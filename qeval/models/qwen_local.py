from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    value = str(dtype_name).lower()
    if value in {"auto", "none"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[value]


def _load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record["sample_id"]
            records[sample_id] = record
    return records


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _chunk_list(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


class QwenLocalGenerator:
    def __init__(self, protocol_config: dict[str, Any]) -> None:
        self.protocol_config = protocol_config
        self.runtime_config = protocol_config.get("runtime", {})
        self.generation_config = protocol_config.get("generation", {})
        self.model_path = protocol_config["target_model"]["path"]

        self.batch_size = int(self.generation_config.get("batch_size", 4))
        self.max_input_tokens = int(self.generation_config.get("max_input_tokens", 2048))
        self.max_new_tokens = int(self.generation_config.get("max_new_tokens", 256))
        self.do_sample = bool(self.generation_config.get("do_sample", False))
        self.temperature = float(self.generation_config.get("temperature", 0.0))
        self.top_p = float(self.generation_config.get("top_p", 1.0))
        self.top_k = int(self.generation_config.get("top_k", 0))
        self.repetition_penalty = float(self.generation_config.get("repetition_penalty", 1.0))
        self.sample_cache = bool(self.runtime_config.get("sample_cache", True))

        torch_dtype = _resolve_torch_dtype(self.runtime_config.get("torch_dtype"))
        device_map = self.runtime_config.get("device_map", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {prompt}\nAssistant:"

    def _generate_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [self._format_chat_prompt(record["prompt"]) for record in batch]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": bool(self.generation_config.get("use_cache", True)),
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                generation_kwargs["top_k"] = self.top_k

        with torch.inference_mode():
            outputs = self.model.generate(**encoded, **generation_kwargs)

        prompt_token_length = encoded["input_ids"].shape[1]
        generated = outputs[:, prompt_token_length:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        predictions: list[dict[str, Any]] = []
        for sample, generation in zip(batch, decoded):
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_dataset": sample["source_dataset"],
                    "split": sample["split"],
                    "prompt": sample["prompt"],
                    "answer": sample.get("answer"),
                    "label": sample.get("label"),
                    "meta": sample.get("meta", {}),
                    "generation": generation.strip(),
                }
            )
        return predictions

    def generate_records(self, records: list[dict[str, Any]], predictions_path: str | Path) -> list[dict[str, Any]]:
        predictions_path = Path(predictions_path)
        cached = _load_jsonl_map(predictions_path) if self.sample_cache else {}
        missing = [record for record in records if record["sample_id"] not in cached]

        prediction_map = dict(cached)
        batches = _chunk_list(missing, self.batch_size)
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            batch_predictions = self._generate_batch(batch)
            _append_jsonl(predictions_path, batch_predictions)
            for record in batch_predictions:
                prediction_map[record["sample_id"]] = record
            print(
                f"[qwen_local] wrote batch {batch_index}/{total_batches} "
                f"({len(batch_predictions)} samples) -> {predictions_path}",
                flush=True,
            )

        return [prediction_map[record["sample_id"]] for record in records]
