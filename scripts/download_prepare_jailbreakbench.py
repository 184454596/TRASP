#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET_REPO = "JailbreakBench/JBB-Behaviors"
DEFAULT_CONFIG = "behaviors"
DEFAULT_SPLITS = "harmful,benign"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_ETAG_TIMEOUT = 60
DEFAULT_HF_DOWNLOAD_TIMEOUT = 120
DEFAULT_RAW_ROOT = "/home/Wangjl/TRASP/data/raw/jailbreakbench"
DEFAULT_MANIFESTS_DIR = "/home/Wangjl/TRASP/data/processed/manifests"


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-") or "default"


def config_tag(config: str) -> str:
    return f"jailbreakbench_{sanitize_name(config)}"


def parse_splits_arg(text: str):
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one split must be provided.")
    return parts


def split_to_label(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "harmful":
        return "harmful"
    if normalized == "benign":
        return "safe"
    raise ValueError(
        f"Unsupported split '{split}'. This script currently expects JailbreakBench 'harmful' and 'benign' splits."
    )


def build_meta(row: dict, dataset_repo: str, config: str, split: str, sample_index: int) -> dict:
    return {
        "dataset_repo": dataset_repo,
        "config": config,
        "original_split": split,
        "jbb_sample_index": sample_index,
        "behavior": row.get("Behavior"),
        "goal": row.get("Goal"),
        "target": row.get("Target"),
        "category": row.get("Category"),
        "source": row.get("Source"),
        "original_row": row,
    }


def build_record(row: dict, dataset_repo: str, config: str, processed_split: str, original_split: str, sample_index: int) -> dict:
    tag = config_tag(config)
    return {
        "sample_id": f"{tag}::{original_split}::{sample_index:06d}",
        "source_dataset": "jailbreakbench",
        "split": processed_split,
        "prompt": row["Goal"],
        "answer": None,
        "label": split_to_label(original_split),
        "meta": build_meta(row, dataset_repo, config, original_split, sample_index),
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_hf_mirror(hf_endpoint: str, etag_timeout: int, download_timeout: int):
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)


def load_jailbreakbench_dataset(dataset_repo: str, config: str, cache_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the Hugging Face 'datasets' package. "
            "Install dependencies first, e.g. `pip install -r /home/Wangjl/TRASP/requirements.txt`."
        ) from exc

    return load_dataset(dataset_repo, config, cache_dir=str(cache_dir))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download JailbreakBench from Hugging Face via mirror and convert it into repo-compatible manifests."
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO, help="Hugging Face dataset repo.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name. Default: behaviors")
    parser.add_argument(
        "--splits",
        default=DEFAULT_SPLITS,
        help="Comma-separated splits to process. Default: harmful,benign",
    )
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT, help="HF endpoint to use, e.g. https://hf-mirror.com")
    parser.add_argument("--hf-etag-timeout", type=int, default=DEFAULT_HF_ETAG_TIMEOUT, help="HF metadata timeout seconds.")
    parser.add_argument(
        "--hf-download-timeout",
        type=int,
        default=DEFAULT_HF_DOWNLOAD_TIMEOUT,
        help="HF download timeout seconds.",
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Directory for raw downloaded/converted data.")
    parser.add_argument("--manifests-dir", default=DEFAULT_MANIFESTS_DIR, help="Directory for processed manifests.")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional limit per split, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw and processed outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    splits = parse_splits_arg(args.splits)

    configure_hf_mirror(args.hf_endpoint, args.hf_etag_timeout, args.hf_download_timeout)

    raw_root = Path(args.raw_root)
    manifests_dir = Path(args.manifests_dir)
    cache_dir = raw_root / ".hf_cache"
    config_name = sanitize_name(args.config)
    dataset_tag = config_tag(args.config)
    raw_config_dir = raw_root / config_name

    print(f"[Info] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"[Info] HF_HUB_ETAG_TIMEOUT={os.environ['HF_HUB_ETAG_TIMEOUT']}")
    print(f"[Info] HF_HUB_DOWNLOAD_TIMEOUT={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}")

    dataset = load_jailbreakbench_dataset(args.dataset_repo, args.config, cache_dir)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": args.dataset_repo,
        "config": args.config,
        "requested_splits": splits,
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "hf_hub_etag_timeout": int(os.environ["HF_HUB_ETAG_TIMEOUT"]),
        "hf_hub_download_timeout": int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]),
        "raw_root": str(raw_config_dir),
        "manifests_dir": str(manifests_dir),
        "outputs": {},
    }

    combined_records = []
    expected_raw_paths = []
    for split in splits:
        raw_path = raw_config_dir / f"{split}.jsonl"
        expected_raw_paths.append(raw_path)

    all_path = manifests_dir / f"{dataset_tag}_all.jsonl"
    harmful_path = manifests_dir / f"harmful_eval_{dataset_tag}.jsonl"
    safe_path = manifests_dir / f"safe_eval_{dataset_tag}.jsonl"
    summary_path = manifests_dir / f"{dataset_tag}_prepare_summary.json"
    all_expected_paths = expected_raw_paths + [all_path, harmful_path, safe_path, summary_path]
    if not args.overwrite:
        conflicts = [str(path) for path in all_expected_paths if path.exists()]
        if conflicts:
            raise FileExistsError(
                "Refusing to overwrite existing files. Use --overwrite to replace them:\n"
                + "\n".join(conflicts)
            )

    for split in splits:
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

        rows = list(dataset[split])
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]

        raw_rows = []
        split_records = []
        for idx, row in enumerate(rows):
            raw_row = dict(row)
            raw_row["jbb_sample_index"] = idx
            raw_row["original_split"] = split
            raw_rows.append(raw_row)
            split_records.append(build_record(row, args.dataset_repo, args.config, f"{dataset_tag}_{split}", split, idx))

        raw_path = raw_config_dir / f"{split}.jsonl"
        write_jsonl(raw_path, raw_rows)

        combined_records.extend(split_records)

        summary["outputs"][split] = {
            "raw_path": str(raw_path),
            "num_raw_rows": len(raw_rows),
            "num_records": len(split_records),
            "label": split_to_label(split),
        }

        print(f"[Done] Split={split}")
        print(f"  raw:    {raw_path}")
        print(f"  count:  {len(split_records)}")

    for record in combined_records:
        record["split"] = f"{dataset_tag}_all"

    harmful_records = []
    safe_records = []
    for record in combined_records:
        copied = dict(record)
        if record["label"] == "harmful":
            copied["split"] = f"harmful_eval_{dataset_tag}"
            harmful_records.append(copied)
        elif record["label"] == "safe":
            copied["split"] = f"safe_eval_{dataset_tag}"
            safe_records.append(copied)

    write_jsonl(all_path, combined_records)
    write_jsonl(harmful_path, harmful_records)
    write_jsonl(safe_path, safe_records)

    summary["outputs"]["combined"] = {
        "all_manifest_path": str(all_path),
        "harmful_manifest_path": str(harmful_path),
        "safe_manifest_path": str(safe_path),
        "num_all_records": len(combined_records),
        "num_harmful_records": len(harmful_records),
        "num_safe_records": len(safe_records),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Combined]")
    print(f"  all:      {all_path}")
    print(f"  harmful:  {harmful_path}")
    print(f"  safe:     {safe_path}")
    print(f"  counts:   total={len(combined_records)} harmful={len(harmful_records)} safe={len(safe_records)}")
    print(f"\n[Summary] {summary_path}")


if __name__ == "__main__":
    main()
