import dataclasses
import functools
import logging
import platform
import shutil
from typing import Any

from PIL import Image, ImageDraw

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.tokenizer as _pi_tokenizer
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def wandb_batch_sample_panel(obs: _model.Observation, actions: _model.Actions | None, batch_idx: int, max_token_len: int) -> wandb.Image:
    obs_h = jax.device_get(obs)
    prompt_txt = ""
    if obs_h.tokenized_prompt is not None:
        tp = np.asarray(obs_h.tokenized_prompt[batch_idx])
        tm = (
            np.asarray(obs_h.tokenized_prompt_mask[batch_idx], dtype=bool)
            if obs_h.tokenized_prompt_mask is not None
            else None
        )
        valid = tp[tm] if tm is not None else tp
        if valid.size:
            try:
                sp = _pi_tokenizer.PaligemmaTokenizer(max_token_len)._tokenizer
                prompt_txt = sp.decode(valid.astype(np.int32).tolist())
            except Exception:
                prompt_txt = "<decode failed>"
    if len(prompt_txt) > 240:
        prompt_txt = prompt_txt[:237] + "..."
    panels, names = [], []
    for name, arr in obs_h.images.items():
        x = np.asarray(arr[batch_idx])
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
        panels.append(Image.fromarray(x))
        names.append(name)
    mh = max(p.height for p in panels)
    prompt_h = 36
    padded = []
    for p in panels:
        if p.height < mh:
            bg = Image.new("RGB", (p.width, mh), (32, 32, 32))
            bg.paste(p, (0, (mh - p.height) // 2))
            p = bg
        padded.append(p)
    # Overlay camera name on each image so it's impossible to miss (catches camera-swap bugs).
    for p, lab in zip(padded, names):
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, p.width, 20], fill=(0, 0, 0))
        d.text((4, 4), lab, fill=(255, 220, 0))
    # Action stats line (catches wrong data ranges / mis-assigned action dims).
    action_txt = ""
    if actions is not None:
        act = np.asarray(jax.device_get(actions))[batch_idx]
        act_flat = act.reshape(-1, act.shape[-1])
        min_dim = int(act_flat.min(axis=0).argmin())
        max_dim = int(act_flat.max(axis=0).argmax())
        action_txt = (
            f"actions  shape={act.shape}  min={act.min():.3f}(d{min_dim})  "
            f"max={act.max():.3f}(d{max_dim})  mean={act.mean():.3f}  std={act.std():.3f}"
        )
    extra_h = 20 if action_txt else 0
    total_w = sum(p.width for p in padded)
    out = Image.new("RGB", (total_w, prompt_h + mh + extra_h), (20, 20, 20))
    drw = ImageDraw.Draw(out)
    drw.text((6, 6), f"PROMPT: {prompt_txt}", fill=(255, 255, 255))
    x = 0
    for p in padded:
        out.paste(p, (x, prompt_h))
        x += p.width
    if action_txt:
        drw.text((6, prompt_h + mh + 4), action_txt, fill=(180, 180, 255))
    return wandb.Image(out)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, fast_loss: bool
    ):  
        # This enters pi0.py compute_loss()
        if fast_loss:
            ce_masked_loss, flow_chunked_loss = model.compute_loss(rng, observation, actions, train=True, fast_loss=fast_loss)
            flow_loss = jnp.mean(flow_chunked_loss)
            ce_loss = jnp.mean(ce_masked_loss)
            return ce_loss + config.flow_loss_alpha * flow_loss, {"ce_loss": ce_loss, "flow_loss": flow_loss}
        else:
            try:
                chunked_loss = model.compute_loss(rng, observation, actions, train=True, fast_loss=False)
            except TypeError:
                chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            if isinstance(chunked_loss, tuple):
                chunked_loss = chunked_loss[-1]
            return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, fast_loss=config.fast_loss
    )

    params = state.params.filter(config.trainable_filter) # this is where it makes it out of the autodiff 
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    llm_filter = nnx_utils.PathRegex(".*llm.*")
    action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")
    vlm_filter = nnx.All(llm_filter, nnx.Not(action_expert_filter))
    vlm_grad_norm = optax.global_norm(grads.filter(vlm_filter))
    action_expert_grad_norm = optax.global_norm(grads.filter(action_expert_filter))
    info = {
        "loss": loss,
        "ce_loss": aux.get("ce_loss", None),
        "flow_loss": aux.get("flow_loss", None),
        "grad_norm": optax.global_norm(grads),
        "grad_norm_vlm_llm": vlm_grad_norm,
        "grad_norm_action_expert": action_expert_grad_norm,
        "stop_gradient_AE_to_VLM_enabled": float(getattr(config.model, "stop_gradient_AE_to_VLM", False)),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def val_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    obs, acts = batch
    if config.fast_loss:
        ce, flow = model.compute_loss(rng, obs, acts, train=True, fast_loss=True)
        return {"val_loss": jnp.mean(ce) + config.flow_loss_alpha * jnp.mean(flow)}
    out = model.compute_loss(rng, obs, acts, train=True)
    return {"val_loss": jnp.mean(out[-1] if isinstance(out, tuple) else out)}


def _format_metric_for_log(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    checkpoint_base_dir = epath.Path(config.checkpoint_base_dir).expanduser()
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(checkpoint_base_dir).free / 2**30
    assert free_gb >= 50, f"Only {free_gb:.1f}GB free at {config.checkpoint_base_dir}; need >=50GB. Free up space before launching."

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        track_best_checkpoint=config.track_best_checkpoint,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    # CHECKPOINT: Code inserted by AI assistant - Verify final batch after all transforms
    # observation, actions = batch
    # logging.info(f"[TRAIN CHECKPOINT] main: FINAL batch - observation.state shape: {observation.state.shape}, observation.state[:7]: {observation.state[7:]}")
    # logging.info(f"[TRAIN CHECKPOINT] main: FINAL batch - actions shape: {actions.shape}, actions[0, 14:]: {actions[0, 14:]}")
    
    
    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

    
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    
    #=====================
    # # Inspect frozen vs trainable parameters
    # all_params = train_state.params
    # frozen_params = all_params.filter(config.freeze_filter)
    # trainable_params = all_params.filter(config.trainable_filter)
    
    # frozen_flat = list(traverse_util.flatten_dict(frozen_params.to_pure_dict()))
    # trainable_flat = list(traverse_util.flatten_dict(trainable_params.to_pure_dict()))
    # logging.info(f"Frozen parameters: {len(frozen_flat)} parameter groups")
    # logging.info(f"Trainable parameters: {len(trainable_flat)} parameter groups")
    
    # # Log parameter counts and shapes for trainable params
    # total_trainable = sum(v.size for v in trainable_flat if hasattr(v, 'size'))
    # logging.info(f"Total trainable parameters: {total_trainable:,}")
    #=====================

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    val_loader = None
    pval_step = None
    if config.val_interval and config.val_repo_id:
        val_loader = _data_loader.create_val_data_loader(config, sharding=data_sharding)
        pval_step = jax.jit(
            functools.partial(val_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        logging.info(f"Validation enabled: repo={config.val_repo_id}, interval={config.val_interval}, batches={config.num_val_batches}")

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    _dataset_view_log_interval = 33
    for step in pbar:
        if config.wandb_enabled and step % _dataset_view_log_interval == 0:
            _b = int(np.random.default_rng(config.seed + step).integers(0, batch[0].state.shape[0]))
            wandb.log(
                {"dataset_random_view": wandb_batch_sample_panel(batch[0], batch[1], _b, config.model.max_token_len)},
                step=step,
            )
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {k: v for k, v in reduced_info.items() if v is not None}
            info_str = ", ".join(f"{k}={_format_metric_for_log(v)}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        last_val_loss = None
        if pval_step is not None and step > start_step and step % config.val_interval == 0:
            losses = []
            val_iter = iter(val_loader)
            for _ in range(config.num_val_batches):
                vbatch = next(val_iter, None)
                if vbatch is None: break
                with sharding.set_mesh(mesh):
                    losses.append(pval_step(train_rng, train_state, vbatch)["val_loss"])
            last_val_loss = float(jnp.mean(jnp.stack(losses)))
            pbar.write(f"Step {step} VAL: val_loss={last_val_loss:.4f}")
            wandb.log({"val_loss": last_val_loss}, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            metrics = {"val_loss": last_val_loss} if last_val_loss is not None else None
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step, metrics=metrics)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
