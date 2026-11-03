from huggingface_hub import snapshot_download, upload_folder
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, Image
from pathlib import Path
import argparse
import contextlib
import os

from huggingface_hub import HfApi, scan_cache_dir
import shutil

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    HF_LEROBOT_HOME = Path(os.path.expanduser("~/.cache/huggingface/lerobot"))

try:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION as LEROBOT_HUB_REVISION_TAG
except ImportError:
    LEROBOT_HUB_REVISION_TAG = "v2.1"

# Keys that are always Sequence(float32, length=14) when present
VECTOR_KEYS = {
    "observation.state",
    "action",
    "observation.velocity",
    "observation.effort",
}

# Scalar int/float keys
SCALAR_INT_KEYS = {"frame_index", "episode_index", "index", "task_index"}
SCALAR_FLOAT_KEYS = {"timestamp"}


def build_features(table: pa.Table) -> Features:
    """Build Features from the actual columns present in the table."""
    feats = {}

    for name in table.schema.names:
        if name in VECTOR_KEYS:
            # Determine length from actual data
            typ = table.schema.field(name).type
            if pa.types.is_fixed_size_list(typ):
                length = typ.list_size
            else:
                col = table.column(name)
                cell = col[0].as_py() if col.length > 0 else None
                length = len(cell) if cell is not None else None
            feats[name] = Sequence(Value("float32"), length=length) if length else Sequence(Value("float32"))

        elif name.startswith("observation.images."):
            feats[name] = Image()

        elif name in SCALAR_INT_KEYS:
            feats[name] = Value("int64")

        elif name in SCALAR_FLOAT_KEYS:
            feats[name] = Value("float32")

        else:
            # Infer from Arrow type
            typ = table.schema.field(name).type
            if pa.types.is_int32(typ) or pa.types.is_int64(typ):
                feats[name] = Value("int64")
            elif pa.types.is_float32(typ):
                feats[name] = Value("float32")
            elif pa.types.is_float64(typ):
                feats[name] = Value("float64")
            elif pa.types.is_string(typ) or pa.types.is_large_string(typ):
                feats[name] = Value("string")
            else:
                feats[name] = Value("float32")

    return Features(feats)


def fix_file(path):
    print(f"Processing {path}")
    table = pq.read_table(path)
    table = table.cast(table.schema.remove_metadata())

    feats = build_features(table)
    ds = Dataset(table)
    ds = ds.cast(feats)

    for k in ["observation.state", "action"]:
        if k in ds.features:
            assert ds.features[k]._type == "Sequence", f"{k} is not Sequence"

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        ds.to_parquet(str(tmp))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def purge_local_caches(repo_id: str) -> None:
    try:
        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "dataset":
                hashes = [r.commit_hash for r in repo.revisions]
                if hashes:
                    info.delete_revisions(*hashes).execute()
                    print(f"Cleared huggingface_hub cache for {repo_id} ({len(hashes)} revision(s)).")
                break
        else:
            print(f"No huggingface_hub cache entry found for {repo_id}.")
    except Exception as e:
        print(f"Warning: huggingface_hub cache purge failed: {e}")

    lr_root = Path(HF_LEROBOT_HOME) / repo_id
    try:
        if lr_root.is_dir():
            shutil.rmtree(lr_root)
            print(f"Removed LeRobot dataset cache: {lr_root}")
    except Exception as e:
        print(f"Warning: LeRobot cache purge failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_repo_id", type=str, required=True, help="HF dataset repo id")
    parser.add_argument("--no-upload", action="store_true", help="Fix local snapshot only; no Hub upload.")
    parser.add_argument("--keep-local-cache", action="store_true", help="Don't delete local caches after upload.")
    parser.add_argument(
        "--local-root",
        action="store_true",
        help="Copy the fixed snapshot to $HF_LEROBOT_HOME/<repo_id> so training reads it "
        "without touching the Hub. Implies --no-upload.",
    )
    args = parser.parse_args()
    repo_id = args.src_repo_id

    local_path = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

    for p in sorted(local_path.rglob("*.parquet")):
        fix_file(p)

    if args.local_root:
        dest = HF_LEROBOT_HOME / repo_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_path, dest)
        print(f"Copied fixed dataset to {dest} (no Hub upload).")
        return

    if args.no_upload:
        print("Skipping Hub upload (--no-upload).")
        return

    upload_folder(repo_id=repo_id, folder_path=str(local_path), repo_type="dataset")

    api = HfApi()
    tag = LEROBOT_HUB_REVISION_TAG
    with contextlib.suppress(Exception):
        api.delete_tag(repo_id, tag=tag, repo_type="dataset")
    api.create_tag(repo_id, tag=tag, repo_type="dataset")
    print(f"Uploaded fixed Parquet; Hub tag {tag!r} now points at this revision.")

    if not args.keep_local_cache:
        purge_local_caches(repo_id)
    else:
        print("Keeping local caches (--keep-local-cache).")


if __name__ == "__main__":
    main()
