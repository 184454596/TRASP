from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


NUMERIC_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_answer(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("$", "").replace(",", "").strip()
    cleaned = cleaned.rstrip(".")
    try:
        normalized = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized.normalize(), "f").rstrip("0").rstrip(".")


def extract_final_numeric_answer(generation: str) -> str | None:
    text = generation.strip()
    if not text:
        return None

    candidate_sections = [text]
    for marker in ["####", "final answer", "answer is", "therefore", "so the answer is"]:
        marker_index = text.lower().rfind(marker)
        if marker_index != -1:
            candidate_sections.insert(0, text[marker_index + len(marker) :])

    for section in candidate_sections:
        matches = NUMERIC_PATTERN.findall(section)
        if matches:
            return normalize_numeric_answer(matches[-1])
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class GSM8KEvaluator:
    def evaluate(
        self,
        samples: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        results_path: str | Path,
    ) -> dict[str, Any]:
        prediction_map = {row["sample_id"]: row for row in predictions}

        total = len(samples)
        correct = 0
        extraction_failures = 0
        rows: list[dict[str, Any]] = []

        for sample in samples:
            prediction = prediction_map[sample["sample_id"]]
            extracted = extract_final_numeric_answer(prediction["generation"])
            gold = normalize_numeric_answer(sample.get("answer"))
            is_correct = extracted is not None and gold is not None and extracted == gold
            extraction_failed = extracted is None

            if is_correct:
                correct += 1
            if extraction_failed:
                extraction_failures += 1

            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "prompt": sample["prompt"],
                    "gold_answer": gold,
                    "generation": prediction["generation"],
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "extraction_failed": extraction_failed,
                }
            )

        write_jsonl(Path(results_path), rows)

        accuracy = correct / total if total else 0.0
        extraction_failure_rate = extraction_failures / total if total else 0.0
        return {
            "accuracy_em": accuracy,
            "extraction_failure_rate": extraction_failure_rate,
            "counts": {
                "num_samples": total,
                "num_correct": correct,
                "num_extraction_failures": extraction_failures,
            },
            "sample_results_path": str(Path(results_path)),
        }
