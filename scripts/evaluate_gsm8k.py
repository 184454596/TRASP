#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_DATA_PATH = "/home/Wangjl/TRASP/data/processed/manifests/gsm8k_test.jsonl"
DEFAULT_MODEL_PATH = "/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = "/home/Wangjl/TRASP/outputs/gsm8k"


def load_samples(data_path: Path):
    with data_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_samples(samples, start_index: int, limit):
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if start_index >= len(samples):
        raise IndexError(f"start_index {start_index} out of range for dataset of size {len(samples)}")

    end_index = len(samples) if limit is None else min(len(samples), start_index + limit)
    return samples[start_index:end_index]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-") or "run"


def extract_answer(text: str):
    text = text.strip()

    match = re.search(r"####\s*\\?[$€£¥]?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1)

    match = re.search(
        r"(?:final answer|answer|答案(?:是|为)?)\s*[:：]?\s*\\?[$€£¥]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)

    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_explicit_answer(text: str):
    text = text.strip()

    match = re.search(r"####\s*\\?[$€£¥]?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1), "hash"

    match = re.search(
        r"(?:final answer|answer|答案(?:是|为)?)\s*[:：]?\s*\\?[$€£¥]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1), "answer_marker"

    return None, None


def normalize_num(text):
    if text is None:
        return None
    text = text.replace(",", "").strip()
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def judge_prediction(pred_text: str, gold_text: str, allow_last_number_fallback: bool = False):
    extracted, parse_mode = extract_explicit_answer(pred_text)
    if extracted is None and allow_last_number_fallback:
        extracted = extract_answer(pred_text)
        if extracted is not None:
            parse_mode = "last_number_fallback"

    pred = normalize_num(extracted)
    gold = normalize_num(gold_text)
    correct = pred == gold and pred is not None
    return {
        "has_explicit_final_answer": parse_mode in {"hash", "answer_marker"},
        "parse_mode": parse_mode,
        "extracted_prediction": extracted,
        "normalized_prediction": str(pred) if pred is not None else None,
        "normalized_gold": str(gold) if gold is not None else None,
        "correct": correct,
    }


def build_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem.\n"
        "Keep the reasoning concise and do not repeat the question.\n"
        "The last line must be exactly in the format:\n"
        "#### <answer>\n\n"
        f"Question: {question}"
    )


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def build_output_dir(output_root: Path, model_path: Path, split_name: str, run_name):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{sanitize_name(model_path.name)}_{sanitize_name(split_name)}_{timestamp}"
    return output_root / run_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a local causal LM on GSM8K and compute accuracy."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to gsm8k jsonl file.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to local model.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Directory where outputs are stored.")
    parser.add_argument("--run-name", default=None, help="Optional output subdirectory name.")
    parser.add_argument("--start-index", type=int, default=0, help="Start sample index, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--allow-last-number-fallback",
        action="store_true",
        help="If no explicit final answer is found, fall back to the last number in the output.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite predictions and summary files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be > 0 when provided")

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_root = Path(args.output_root)

    all_samples = load_samples(data_path)
    selected_samples = select_samples(all_samples, args.start_index, args.limit)
    split_name = selected_samples[0]["split"] if selected_samples else "gsm8k"

    output_dir = build_output_dir(output_root, model_path, split_name, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (predictions_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Output files already exist in {output_dir}. "
            "Pass --overwrite or use a different --run-name."
        )

    print(f"[Info] Data path: {data_path}")
    print(f"[Info] Model path: {model_path}")
    print(f"[Info] Dataset size: {len(all_samples)}")
    print(f"[Info] Evaluating samples: {len(selected_samples)}")
    print(f"[Info] Start index: {args.start_index}")
    print(f"[Info] Batch size: {args.batch_size}")
    print(f"[Info] Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    input_device = get_input_device(model)
    prompt_template = [{"role": "user", "content": None}]

    correct_count = 0
    invalid_prediction_count = 0
    missing_explicit_answer_count = 0
    evaluated_count = 0

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
    else:
        generate_kwargs["do_sample"] = False

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for batch_start, batch_samples in enumerate(
            tqdm(batched(selected_samples, args.batch_size), total=(len(selected_samples) + args.batch_size - 1) // args.batch_size, desc="Evaluating"),
            start=0,
        ):
            prompts = [build_prompt(sample["prompt"]) for sample in batch_samples]
            batch_texts = []
            for prompt in prompts:
                messages = [dict(prompt_template[0])]
                messages[0]["content"] = prompt
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_texts.append(text)

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
            ).to(input_device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **generate_kwargs)

            generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for offset, (sample, response) in enumerate(zip(batch_samples, responses)):
                judgement = judge_prediction(
                    response,
                    sample["answer"],
                    allow_last_number_fallback=args.allow_last_number_fallback,
                )
                correct_count += int(judgement["correct"])
                invalid_prediction_count += int(judgement["normalized_prediction"] is None)
                missing_explicit_answer_count += int(not judgement["has_explicit_final_answer"])
                evaluated_count += 1

                record = {
                    "sample_index": args.start_index + batch_start * args.batch_size + offset,
                    "sample_id": sample["sample_id"],
                    "split": sample["split"],
                    "source_dataset": sample["source_dataset"],
                    "question": sample["prompt"],
                    "gold_answer": sample["answer"],
                    "model_output": response.strip(),
                    **judgement,
                }
                pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            pred_f.flush()

    accuracy = correct_count / evaluated_count if evaluated_count else 0.0
    summary = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "dataset_total_count": len(all_samples),
        "evaluated_count": evaluated_count,
        "start_index": args.start_index,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "correct_count": correct_count,
        "invalid_prediction_count": invalid_prediction_count,
        "missing_explicit_answer_count": missing_explicit_answer_count,
        "accuracy": accuracy,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
