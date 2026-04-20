#!/usr/bin/env python3
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


DEFAULT_DATA_PATH = "/home/Wangjl/TRASP/data/processed/manifests/gsm8k_test.jsonl"
DEFAULT_MODEL_PATH = "/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct"


def load_sample(data_path: Path, sample_index: int) -> dict:
    with data_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index {sample_index} out of range for {data_path}")


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


def is_correct(pred_text: str, gold_text: str):
    pred = normalize_num(extract_answer(pred_text))
    gold = normalize_num(gold_text)
    return pred == gold and pred is not None, pred, gold


def build_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem.\n"
        "The last line must be exactly in the format:\n"
        "#### <answer>\n\n"
        f"Question: {question}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-7B-Instruct on one GSM8K sample and judge correctness."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to gsm8k jsonl file.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to local model.")
    parser.add_argument("--sample-index", type=int, default=0, help="0-based sample index.")
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
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    sample = load_sample(data_path, args.sample_index)
    prompt = build_prompt(sample["prompt"])

    print(f"[Info] Data path: {data_path}")
    print(f"[Info] Model path: {model_path}")
    print(f"[Info] Sample index: {args.sample_index}")
    print(f"[Info] Sample id: {sample['sample_id']}")
    print("\n=== Question ===")
    print(sample["prompt"])

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

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    explicit_answer, parse_mode = extract_explicit_answer(response)
    extracted = explicit_answer
    if extracted is None and args.allow_last_number_fallback:
        extracted = extract_answer(response)

    pred_num = normalize_num(extracted)
    gold_num = normalize_num(sample["answer"])
    correct = pred_num == gold_num and pred_num is not None

    print("\n=== Model Output ===")
    print(response)
    print("\n=== Judgement ===")
    print(f"Explicit final answer found: {explicit_answer is not None}")
    print(f"Parse mode: {parse_mode}")
    print(f"Extracted prediction: {extracted}")
    print(f"Normalized prediction: {pred_num}")
    print(f"Gold answer: {sample['answer']}")
    print(f"Normalized gold: {gold_num}")
    print(f"Correct: {correct}")


if __name__ == "__main__":
    main()
