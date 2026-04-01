from __future__ import annotations

import argparse
import json
import platform
import random
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from qeval.common.config import (
    build_resolved_config,
    get_manifest_paths,
    get_output_root,
    load_protocol_bundle,
)
from qeval.eval.gsm8k_eval import GSM8KEvaluator
from qeval.eval.harmbench_judge import HarmBenchJudge
from qeval.eval.xstest_eval import XSTestEvaluator
from qeval.models.qwen_local import QwenLocalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base model clean evaluation for the fixed Qwen protocol.")
    parser.add_argument(
        "--config",
        required=True,
        help="Absolute path to the protocol YAML.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the output run directory name.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def collect_env() -> dict[str, Any]:
    import transformers

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Clean Evaluation Summary",
        "",
        f"- clean_asr: {metrics['clean_asr']:.6f}",
        f"- clean_utility_em: {metrics['clean_utility_em']:.6f}",
        f"- clean_benign_refusal: {metrics['clean_benign_refusal']:.6f}",
        "",
        "## Counts",
        "",
        f"- harmful_counts: {json.dumps(metrics['harmful_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- gsm8k_counts: {json.dumps(metrics['gsm8k_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- xstest_counts: {json.dumps(metrics['xstest_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Per-dataset ASR",
        "",
        f"- per_dataset_asr: {json.dumps(metrics['per_dataset_asr'], ensure_ascii=False, sort_keys=True)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_harmful_metrics(judgements: list[dict[str, Any]]) -> tuple[float, dict[str, float], dict[str, Any]]:
    total = len(judgements)
    successes = sum(1 for row in judgements if row["is_attack_success"])
    parse_failures = sum(1 for row in judgements if not row["judge_parse_success"])

    per_dataset_asr: dict[str, float] = {}
    for dataset_name in sorted({row["source_dataset"] for row in judgements}):
        subset = [row for row in judgements if row["source_dataset"] == dataset_name]
        subset_total = len(subset)
        subset_successes = sum(1 for row in subset if row["is_attack_success"])
        per_dataset_asr[dataset_name] = subset_successes / subset_total if subset_total else 0.0

    return (
        successes / total if total else 0.0,
        per_dataset_asr,
        {
            "num_samples": total,
            "num_attack_successes": successes,
            "num_parse_failures": parse_failures,
        },
    )


def main() -> int:
    args = parse_args()
    bundle = load_protocol_bundle(args.config)
    resolved_config = build_resolved_config(bundle)
    protocol = bundle["protocol"]
    runtime = protocol.get("runtime", {})
    seed = int(runtime.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    run_name = args.run_name or runtime.get("run_name", "clean_base_qwen_eval")
    output_root = get_output_root(bundle)
    run_dir = output_root / run_name
    predictions_dir = run_dir / "predictions"
    judgements_dir = run_dir / "judgements"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    judgements_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, allow_unicode=True, sort_keys=False)
    write_json(run_dir / "env.json", collect_env())

    manifest_paths = get_manifest_paths(bundle)

    harmful_samples = load_jsonl(manifest_paths["harmful_eval_unified"])
    print(f"[run_clean_eval] harmful stage: {len(harmful_samples)} samples", flush=True)
    qwen = QwenLocalGenerator(protocol)
    harmful_predictions = qwen.generate_records(
        harmful_samples,
        predictions_dir / "harmful_eval_unified_predictions.jsonl",
    )
    qwen.close()

    judge = HarmBenchJudge(bundle["judges"])
    harmful_judgements = judge.judge_records(
        harmful_samples,
        harmful_predictions,
        judgements_dir / "harmful_eval_unified_judgements.jsonl",
    )
    judge.close()
    print("[run_clean_eval] harmful stage complete", flush=True)

    clean_asr, per_dataset_asr, harmful_counts = compute_harmful_metrics(harmful_judgements)

    qwen = QwenLocalGenerator(protocol)
    gsm8k_samples = load_jsonl(manifest_paths["gsm8k_test"])
    print(f"[run_clean_eval] gsm8k stage: {len(gsm8k_samples)} samples", flush=True)
    gsm8k_predictions = qwen.generate_records(
        gsm8k_samples,
        predictions_dir / "gsm8k_test_predictions.jsonl",
    )
    gsm8k_metrics = GSM8KEvaluator().evaluate(
        gsm8k_samples,
        gsm8k_predictions,
        judgements_dir / "gsm8k_test_results.jsonl",
    )
    print("[run_clean_eval] gsm8k stage complete", flush=True)

    xstest_samples = load_jsonl(manifest_paths["xstest_safe_eval"])
    print(f"[run_clean_eval] xstest stage: {len(xstest_samples)} samples", flush=True)
    xstest_predictions = qwen.generate_records(
        xstest_samples,
        predictions_dir / "xstest_safe_eval_predictions.jsonl",
    )
    qwen.close()

    xstest_metrics = XSTestEvaluator().evaluate(
        xstest_samples,
        xstest_predictions,
        judgements_dir / "xstest_safe_eval_results.jsonl",
    )
    print("[run_clean_eval] xstest stage complete", flush=True)

    metrics = {
        "run_name": run_name,
        "clean_asr": clean_asr,
        "clean_utility_em": gsm8k_metrics["accuracy_em"],
        "clean_benign_refusal": xstest_metrics["refusal_rate"],
        "harmful_counts": harmful_counts,
        "gsm8k_counts": gsm8k_metrics["counts"],
        "xstest_counts": xstest_metrics["counts"],
        "per_dataset_asr": per_dataset_asr,
        "gsm8k_extraction_failure_rate": gsm8k_metrics["extraction_failure_rate"],
        "artifacts": {
            "harmful_predictions": str(predictions_dir / "harmful_eval_unified_predictions.jsonl"),
            "harmful_judgements": str(judgements_dir / "harmful_eval_unified_judgements.jsonl"),
            "gsm8k_predictions": str(predictions_dir / "gsm8k_test_predictions.jsonl"),
            "gsm8k_results": str(judgements_dir / "gsm8k_test_results.jsonl"),
            "xstest_predictions": str(predictions_dir / "xstest_safe_eval_predictions.jsonl"),
            "xstest_results": str(judgements_dir / "xstest_safe_eval_results.jsonl"),
        },
    }

    write_json(run_dir / "metrics.json", metrics)
    write_summary(run_dir / "summary.md", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
