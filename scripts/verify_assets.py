from __future__ import annotations

import json
from pathlib import Path
import sys


QWEN_MODEL_DIR = Path("/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct")
HARMBENCH_CLASSIFIER_DIR = Path("/home/Wangjl/TRASP/models/HarmBench-Llama-2-13b-cls")
GSM8K_DIR = Path("/home/Wangjl/TRASP/data/raw/gsm8k")
XSTEST_DIR = Path("/home/Wangjl/TRASP/data/raw/xstest")
ADVBENCH_DIR = Path("/home/Wangjl/TRASP/data/raw/advbench")
HARMBENCH_REPO_DIR = Path("/home/Wangjl/TRASP/external/HarmBench")
HARMBENCH_BEHAVIOR_DIR = HARMBENCH_REPO_DIR / "data" / "behavior_datasets"


def check_dir(label: str, path: Path) -> bool:
    exists = path.is_dir()
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {label}: {path}")
    return exists


def check_harmbench_classifier(path: Path) -> bool:
    exists = path.is_dir()
    has_config = (path / "config.json").is_file()
    has_tokenizer = (path / "tokenizer_config.json").is_file()
    download_cache_dir = path / ".cache" / "huggingface" / "download"
    has_incomplete = any(download_cache_dir.glob("*.incomplete"))
    has_index = (path / "model.safetensors.index.json").is_file()
    present_shards = sorted(p.name for p in path.glob("model-*.safetensors"))
    has_shards = bool(present_shards)
    has_single_file = (path / "model.safetensors").is_file()
    expected_shards: list[str] = []

    if has_index:
        try:
            data = json.loads((path / "model.safetensors.index.json").read_text())
            expected_shards = sorted(set(data.get("weight_map", {}).values()))
        except Exception:
            expected_shards = []

    missing_shards = [shard for shard in expected_shards if not (path / shard).is_file()]
    incomplete_files = sorted(p.name for p in download_cache_dir.glob("*.incomplete"))

    complete = exists and has_config and has_tokenizer and not has_incomplete and (
        (has_index and has_shards and not missing_shards) or has_single_file
    )
    status = "OK" if complete else "MISSING_OR_INCOMPLETE"
    print(f"[{status}] HarmBench classifier: {path}")
    if not complete:
        if missing_shards:
            print(f"[DETAIL] missing classifier shards: {', '.join(missing_shards)}")
        if incomplete_files:
            print(f"[DETAIL] incomplete download files: {len(incomplete_files)}")
    return complete


def main() -> int:
    checks = [
        ("Qwen local model", QWEN_MODEL_DIR),
        ("GSM8K", GSM8K_DIR),
        ("XSTest", XSTEST_DIR),
        ("AdvBench", ADVBENCH_DIR),
        ("HarmBench repo", HARMBENCH_REPO_DIR),
        ("HarmBench behavior_datasets", HARMBENCH_BEHAVIOR_DIR),
    ]

    all_ok = True
    all_ok = check_harmbench_classifier(HARMBENCH_CLASSIFIER_DIR) and all_ok

    for label, path in checks:
        all_ok = check_dir(label, path) and all_ok

    print(f"[SUMMARY] overall_status={'OK' if all_ok else 'MISSING'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
