#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET_REPO = "PKU-Alignment/PKU-SafeRLHF"
DEFAULT_SUBSET = "default"
DEFAULT_RAW_ROOT = "/home/Wangjl/TRASP/data/raw/pku_saferlhf"
DEFAULT_MANIFESTS_DIR = "/home/Wangjl/TRASP/data/processed/manifests"


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-") or "default"


def subset_tag(subset: str) -> str:
    return "pku_saferlhf" if subset == "default" else f"pku_saferlhf_{sanitize_name(subset)}"


def infer_prompt_label(row: dict) -> str:
    safe0 = bool(row.get("is_response_0_safe", False))
    safe1 = bool(row.get("is_response_1_safe", False))
    return "safe" if safe0 and safe1 else "harmful"


def pick_response(row: dict, response_id):
    if response_id == 0:
        return row.get("response_0")
    if response_id == 1:
        return row.get("response_1")
    return None


def build_meta(row: dict, dataset_repo: str, subset: str, split: str, sample_index: int) -> dict:
    prompt_label = infer_prompt_label(row)
    safe0 = bool(row.get("is_response_0_safe", False))
    safe1 = bool(row.get("is_response_1_safe", False))
    safer_response_id = row.get("safer_response_id")
    better_response_id = row.get("better_response_id")

    return {
        "dataset_repo": dataset_repo,
        "subset": subset,
        "original_split": split,
        "pku_sample_index": sample_index,
        "prompt_source": row.get("prompt_source"),
        "response_0_source": row.get("response_0_source"),
        "response_1_source": row.get("response_1_source"),
        "response_0": row.get("response_0"),
        "response_1": row.get("response_1"),
        "response_0_sha256": row.get("response_0_sha256"),
        "response_1_sha256": row.get("response_1_sha256"),
        "is_response_0_safe": safe0,
        "is_response_1_safe": safe1,
        "response_0_severity_level": row.get("response_0_severity_level"),
        "response_1_severity_level": row.get("response_1_severity_level"),
        "response_0_harm_category": row.get("response_0_harm_category"),
        "response_1_harm_category": row.get("response_1_harm_category"),
        "better_response_id": better_response_id,
        "safer_response_id": safer_response_id,
        "better_response": pick_response(row, better_response_id),
        "safer_response": pick_response(row, safer_response_id),
        "unsafe_response_count": int(not safe0) + int(not safe1),
        "safe_response_count": int(safe0) + int(safe1),
        "prompt_label": prompt_label,
        "prompt_label_rule": "safe iff both annotated responses are safe; otherwise harmful",
    }


def build_record(row: dict, dataset_repo: str, subset: str, processed_split: str, original_split: str, sample_index: int) -> dict:
    label = infer_prompt_label(row)
    subset_name = subset_tag(subset)
    return {
        "sample_id": f"{subset_name}::{original_split}::{sample_index:06d}",
        "source_dataset": "pku_saferlhf",
        "split": processed_split,
        "prompt": row["prompt"],
        "answer": None,
        "label": label,
        "meta": build_meta(row, dataset_repo, subset, original_split, sample_index),
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pku_dataset(dataset_repo: str, subset: str, cache_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the Hugging Face 'datasets' package. "
            "Install dependencies first, e.g. `pip install -r /home/Wangjl/TRASP/requirements.txt`."
        ) from exc

    return load_dataset(dataset_repo, subset, cache_dir=str(cache_dir))


def parse_splits_arg(text: str):
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one split must be provided.")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download PKU-SafeRLHF and convert it into manifest files compatible with this repo."
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO, help="Hugging Face dataset repo.")
    parser.add_argument("--subset", default=DEFAULT_SUBSET, help="Dataset subset/config name. Default: default")
    parser.add_argument("--splits", default="train,test", help="Comma-separated splits to process. Default: train,test")
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

    raw_root = Path(args.raw_root)
    manifests_dir = Path(args.manifests_dir)
    cache_dir = raw_root / ".hf_cache"
    subset_name = subset_tag(args.subset)

    dataset = load_pku_dataset(args.dataset_repo, args.subset, cache_dir)

    raw_subset_dir = raw_root / sanitize_name(args.subset)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo": args.dataset_repo,
        "subset": args.subset,
        "requested_splits": splits,
        "raw_root": str(raw_subset_dir),
        "manifests_dir": str(manifests_dir),
        "prompt_label_rule": "safe iff both annotated responses are safe; otherwise harmful",
        "outputs": {},
    }

    for split in splits:
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")

        rows = list(dataset[split])
        if args.max_samples_per_split is not None:
            rows = rows[: args.max_samples_per_split]

        raw_rows = []
        for idx, row in enumerate(rows):
            raw_row = dict(row)
            raw_row["pku_sample_index"] = idx
            raw_rows.append(raw_row)

        raw_path = raw_subset_dir / f"{split}.jsonl"
        all_path = manifests_dir / f"{subset_name}_{split}_all.jsonl"
        harmful_path = manifests_dir / f"harmful_eval_{subset_name}_{split}.jsonl"
        safe_path = manifests_dir / f"safe_eval_{subset_name}_{split}.jsonl"
        all_split_name = f"{subset_name}_{split}_all"
        harmful_split_name = f"harmful_eval_{subset_name}_{split}"
        safe_split_name = f"safe_eval_{subset_name}_{split}"

        all_records = [
            build_record(row, args.dataset_repo, args.subset, all_split_name, split, idx)
            for idx, row in enumerate(rows)
        ]
        harmful_records = [
            build_record(row, args.dataset_repo, args.subset, harmful_split_name, split, idx)
            for idx, row in enumerate(rows)
            if infer_prompt_label(row) == "harmful"
        ]
        safe_records = [
            build_record(row, args.dataset_repo, args.subset, safe_split_name, split, idx)
            for idx, row in enumerate(rows)
            if infer_prompt_label(row) == "safe"
        ]

        existing_paths = [raw_path, all_path, harmful_path, safe_path]
        if not args.overwrite:
            conflicts = [str(path) for path in existing_paths if path.exists()]
            if conflicts:
                raise FileExistsError(
                    "Refusing to overwrite existing files. Use --overwrite to replace them:\n"
                    + "\n".join(conflicts)
                )

        write_jsonl(raw_path, raw_rows)
        write_jsonl(all_path, all_records)
        write_jsonl(harmful_path, harmful_records)
        write_jsonl(safe_path, safe_records)

        summary["outputs"][split] = {
            "raw_path": str(raw_path),
            "all_manifest_path": str(all_path),
            "harmful_manifest_path": str(harmful_path),
            "safe_manifest_path": str(safe_path),
            "num_raw_rows": len(raw_rows),
            "num_all_records": len(all_records),
            "num_harmful_records": len(harmful_records),
            "num_safe_records": len(safe_records),
        }

        print(f"[Done] Split={split}")
        print(f"  raw:      {raw_path}")
        print(f"  all:      {all_path}")
        print(f"  harmful:  {harmful_path}")
        print(f"  safe:     {safe_path}")
        print(f"  counts:   total={len(all_records)} harmful={len(harmful_records)} safe={len(safe_records)}")

    summary_path = manifests_dir / f"{subset_name}_prepare_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing summary file {summary_path}. Use --overwrite to replace it."
        )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Summary] {summary_path}")


if __name__ == "__main__":
    main()
