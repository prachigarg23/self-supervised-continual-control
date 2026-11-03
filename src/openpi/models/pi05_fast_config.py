import dataclasses
from typing import TYPE_CHECKING, Literal, Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi05_fast import Pi05_FAST


@dataclasses.dataclass(frozen=True)
class Pi05_FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    img_encoder_mode: Literal["freeze", "lora", "trainable"] = "trainable"
    freeze_vlm_params: bool = False
    stop_gradient_AE_to_VLM: bool = False

    # If True, when the VLM is frozen (`freeze_vlm_params=True`), unfreeze its
    # MLP gate+up weights (`gating_einsum` in gemma.py). The MLP down (`linear`)
    # stays frozen. Default False = no behavior change.
    vlm_unfreeze_mlp_gate_up: bool = False

    # If True, when the VLM is frozen, unfreeze the self-attention projections
    # (q/k/v/o = q_einsum, kv_einsum, attn_vec_einsum) in BOTH experts. Default
    # False = no behavior change.
    vlm_unfreeze_sa: bool = False

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05_FAST always uses the Pi0.5 path (adaRMS + time MLP; discrete state token as part of text tokens).
    pi05: bool = True
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    use_sub_task: bool = False

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05_FAST":
        from openpi.models.pi05_fast import Pi05_FAST
        return Pi05_FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        llm = nnx_utils.PathRegex(".*llm.*")
        action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
        img = nnx_utils.PathRegex(".*img.*")
        lora = nnx_utils.PathRegex(".*lora.*")

        freeze_rules = []

        # 1. Freeze VLM backbone if requested (but NOT its LoRA)
        if self.freeze_vlm_params:
            freeze_rules.append(
                nnx.All(llm, nnx.Not(lora), nnx.Not(action_expert))
            )

        # 2. Freeze image encoder if requested
        if self.img_encoder_mode == "freeze":
            freeze_rules.append(img)

        # 3. LoRA on language backbone → freeze its backbone
        if "lora" in self.paligemma_variant:
            freeze_rules.append(
                nnx.All(llm, nnx.Not(lora), nnx.Not(action_expert))
            )
        # import pdb; pdb.set_trace()
        # 4. LoRA on action expert → freeze its backbone
        if "lora" in self.action_expert_variant:
            freeze_rules.append(
                nnx.All(action_expert, nnx.Not(lora), nnx.Not(llm))
            )
        
        if not freeze_rules:
            return nnx.Nothing

        # Freeze if ANY rule matches
        return nnx.Any(*freeze_rules)

    def get_selective_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze both VLM and action-expert backbones, except a chosen subset.

        Flags (mutually exclusive; setting both raises):
          - vlm_unfreeze_mlp_gate_up: keep MLP gate+up (`gating_einsum`) trainable
            in BOTH experts; MLP down (`linear`) stays frozen.
          - vlm_unfreeze_sa: keep self-attention q/k/v/o projections
            (`q_einsum`, `kv_einsum`, `attn_vec_einsum`) trainable in BOTH experts.

        Both recipes also freeze the action-expert backbone (otherwise it would
        already be fully trainable). Default (no flag) returns `get_freeze_filter()`.
        """
        base = self.get_freeze_filter()
        if self.vlm_unfreeze_mlp_gate_up and self.vlm_unfreeze_sa:
            raise ValueError("Set at most one of vlm_unfreeze_mlp_gate_up / vlm_unfreeze_sa.")
        if self.vlm_unfreeze_mlp_gate_up:
            # Matches `mlp/gating_einsum` (VLM) and `mlp_1/gating_einsum` (AE).
            unfreeze = nnx_utils.PathRegex(r".*gating_einsum.*")
        elif self.vlm_unfreeze_sa:
            # Matches q_einsum, kv_einsum, attn_vec_einsum and their `_1` (AE) variants.
            unfreeze = nnx_utils.PathRegex(r".*(q_einsum|kv_einsum|attn_vec_einsum).*")
        else:
            return base

        # Freeze the action-expert backbone too (full fine-tune, no LoRA).
        action_expert = nnx_utils.PathRegex(r".*llm.*_1.*")
        combined_freeze = nnx.Any(base, action_expert)
        return nnx.All(combined_freeze, nnx.Not(unfreeze))


    # def get_freeze_filter(self) -> nnx.filterlib.Filter:
    #     """Returns the freeze filter based on the model config."""
    #     filters = []
    #     has_lora = False
    #     gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
    #     action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
    #     if "lora" in self.paligemma_variant:
    #         filters.append(
    #             gemma_params_filter,
    #         )
    #         if "lora" not in self.action_expert_variant:
    #             # If only freeze gemma params, exclude action expert params.
    #             filters.append(
    #                 nnx.Not(action_expert_params_filter),
    #             )
    #         has_lora = True
    #     elif "lora" in self.action_expert_variant:
    #         filters.append(
    #             action_expert_params_filter,
    #         )
    #         has_lora = True

    #     if has_lora:
    #         # If any lora is used, exclude all lora params.
    #         filters.append(
    #             nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    #         )
    #     if not filters:
    #         return nnx.Nothing
    #     return nnx.All(*filters)
