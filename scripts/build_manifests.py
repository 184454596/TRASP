#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qeval.common.config import get_manifest_paths, get_manifest_summary_path, get_manifests_dir, get_required_fields, load_protocol_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build unified manifests for the base Qwen evaluation protocol.')
    parser.add_argument(
        '--config',
        default='/home/Wangjl/TRASP/configs/protocol/base_qwen_eval.yaml',
        help='Absolute path to the protocol YAML.',
    )
    return parser.parse_args()


def extract_gsm8k_final_answer(raw_answer: str) -> str:
    match = re.search(r'####\s*(.+)', raw_answer)
    if match:
        return match.group(1).strip()
    return raw_answer.strip()


def ensure_required_fields(record: dict[str, Any], required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f'record missing required fields: {missing}')


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def build_harmbench_records(bundle: dict[str, Any], required_fields: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = bundle['datasets']['raw_sources']['harmbench_text']
    preferred_path = Path(config['preferred_path'])
    fallback_path = Path(config['fallback_path'])

    if preferred_path.is_file():
        selected_path = preferred_path
        selection_mode = 'preferred_text_test'
    elif fallback_path.is_file():
        selected_path = fallback_path
        selection_mode = 'fallback_text_all'
    else:
        raise FileNotFoundError('No HarmBench text behavior CSV is available.')

    records: list[dict[str, Any]] = []
    with selected_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            behavior = (row.get('Behavior') or '').strip()
            context = (row.get('ContextString') or '').strip()
            prompt = f'{context}\n\n{behavior}'.strip() if context else behavior
            sample_id_base = (row.get('BehaviorID') or f'{row_idx:06d}').strip()
            record = {
                'sample_id': f'harmbench_text::{sample_id_base}',
                'source_dataset': config['source_dataset'],
                'split': config['output_split'],
                'prompt': prompt,
                'answer': None,
                'label': 'harmful',
                'meta': {
                    'behavior': behavior,
                    'behavior_id': row.get('BehaviorID'),
                    'context_string': context,
                    'functional_category': row.get('FunctionalCategory'),
                    'semantic_category': row.get('SemanticCategory'),
                    'tags': row.get('Tags'),
                    'source_csv': str(selected_path),
                    'selection_mode': selection_mode,
                },
            }
            ensure_required_fields(record, required_fields)
            records.append(record)

    return records, {'selected_csv': str(selected_path), 'selection_mode': selection_mode}


def build_advbench_records(bundle: dict[str, Any], required_fields: list[str]) -> list[dict[str, Any]]:
    config = bundle['datasets']['raw_sources']['advbench']
    dataset = load_from_disk(config['path'])
    split_dataset = dataset[config['split']]
    records: list[dict[str, Any]] = []

    for row_idx, row in enumerate(split_dataset):
        content = row.get('content', '')
        if isinstance(content, list):
            content_text = '\n'.join(str(item).strip() for item in content if str(item).strip())
        else:
            content_text = str(content).strip()
        instructions_text = str(row.get('instructions', '') or '').strip()
        prompt_parts = [part for part in [instructions_text, content_text] if part]
        prompt = '\n\n'.join(prompt_parts).strip()
        record = {
            'sample_id': f'advbench::{row_idx:06d}',
            'source_dataset': config['source_dataset'],
            'split': config['output_split'],
            'prompt': prompt,
            'answer': None,
            'label': 'harmful',
            'meta': {
                'clf_label': row.get('clf_label'),
                'instructions': row.get('instructions'),
                'content': row.get('content'),
                'answer_prompt': row.get('answer_prompt'),
                'proxy_clf_label': row.get('proxy_clf_label'),
                'gen_target': row.get('gen_target'),
                'proxy_gen_target': row.get('proxy_gen_target'),
            },
        }
        ensure_required_fields(record, required_fields)
        records.append(record)

    return records


def build_gsm8k_records(bundle: dict[str, Any], required_fields: list[str], split_name: str) -> list[dict[str, Any]]:
    config = bundle['datasets']['raw_sources']['gsm8k']
    dataset = load_from_disk(config['path'])
    hf_split_name = config[f'{split_name}_split']
    output_split = config['output_splits'][split_name]
    split_dataset = dataset[hf_split_name]
    records: list[dict[str, Any]] = []

    for row_idx, row in enumerate(split_dataset):
        raw_answer = str(row['answer'])
        final_answer = extract_gsm8k_final_answer(raw_answer)
        record = {
            'sample_id': f'gsm8k::{split_name}::{row_idx:06d}',
            'source_dataset': config['source_dataset'],
            'split': output_split,
            'prompt': str(row['question']).strip(),
            'answer': final_answer,
            'label': 'utility',
            'meta': {
                'raw_answer': raw_answer,
                'answer_extraction': 'text_after_####',
            },
        }
        ensure_required_fields(record, required_fields)
        records.append(record)

    return records


def build_xstest_records(bundle: dict[str, Any], required_fields: list[str]) -> list[dict[str, Any]]:
    config = bundle['datasets']['raw_sources']['xstest']
    dataset = load_from_disk(config['path'])
    split_dataset = dataset[config['split']]
    records: list[dict[str, Any]] = []

    for row in split_dataset:
        if str(row.get('label')).strip().lower() != str(config['safe_label_value']).lower():
            continue
        sample_id = int(row.get('id'))
        record = {
            'sample_id': f'xstest::safe::{sample_id:06d}',
            'source_dataset': config['source_dataset'],
            'split': config['output_split'],
            'prompt': str(row['prompt']).strip(),
            'answer': None,
            'label': 'safe',
            'meta': {
                'xstest_id': row.get('id'),
                'type': row.get('type'),
                'focus': row.get('focus'),
                'note': row.get('note'),
                'raw_label': row.get('label'),
            },
        }
        ensure_required_fields(record, required_fields)
        records.append(record)

    return records


def build_harmful_unified_records(harmbench_records: list[dict[str, Any]], advbench_records: list[dict[str, Any]], bundle: dict[str, Any], required_fields: list[str]) -> list[dict[str, Any]]:
    output_split = bundle['datasets']['derived_manifests']['harmful_eval_unified']['output_split']
    combined: list[dict[str, Any]] = []
    for source_records in [harmbench_records, advbench_records]:
        for record in source_records:
            unified = deepcopy(record)
            unified['meta'] = dict(unified['meta'])
            unified['meta']['origin_split'] = unified['split']
            unified['split'] = output_split
            ensure_required_fields(unified, required_fields)
            combined.append(unified)
    return combined


def build_summary(bundle: dict[str, Any], required_fields: list[str], manifests: dict[str, list[dict[str, Any]]], harmbench_info: dict[str, Any]) -> dict[str, Any]:
    manifest_paths = get_manifest_paths(bundle)
    summary: dict[str, Any] = {
        'protocol_name': bundle['protocol']['protocol_name'],
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifests_dir': str(get_manifests_dir(bundle)),
        'required_fields': required_fields,
        'field_types': {
            'sample_id': 'string',
            'source_dataset': 'string',
            'split': 'string',
            'prompt': 'string',
            'answer': 'string_or_null',
            'label': 'string',
            'meta': 'object',
        },
        'harmbench_text_selection': harmbench_info,
        'manifests': {},
    }

    for manifest_name, records in manifests.items():
        path = manifest_paths[manifest_name]
        summary['manifests'][manifest_name] = {
            'path': str(path),
            'num_samples': len(records),
            'source_datasets': sorted({record['source_dataset'] for record in records}),
            'field_names': required_fields,
        }

    return summary


def main() -> int:
    args = parse_args()
    bundle = load_protocol_bundle(args.config)
    required_fields = get_required_fields(bundle)
    manifests_dir = get_manifests_dir(bundle)
    summary_path = get_manifest_summary_path(bundle)
    manifest_paths = get_manifest_paths(bundle)

    manifests_dir.mkdir(parents=True, exist_ok=True)

    harmbench_records, harmbench_info = build_harmbench_records(bundle, required_fields)
    advbench_records = build_advbench_records(bundle, required_fields)
    gsm8k_train_records = build_gsm8k_records(bundle, required_fields, 'train')
    gsm8k_test_records = build_gsm8k_records(bundle, required_fields, 'test')
    xstest_safe_records = build_xstest_records(bundle, required_fields)
    harmful_unified_records = build_harmful_unified_records(harmbench_records, advbench_records, bundle, required_fields)

    manifests: dict[str, list[dict[str, Any]]] = {
        'harmful_eval_harmbench_text': harmbench_records,
        'harmful_eval_advbench': advbench_records,
        'harmful_eval_unified': harmful_unified_records,
        'gsm8k_train': gsm8k_train_records,
        'gsm8k_test': gsm8k_test_records,
        'xstest_safe_eval': xstest_safe_records,
    }

    for manifest_name, records in manifests.items():
        path = manifest_paths[manifest_name]
        write_jsonl(path, records)
        print(f'[build_manifests] wrote {manifest_name}: {len(records)} -> {path}')

    summary = build_summary(bundle, required_fields, manifests, harmbench_info)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'[build_manifests] wrote summary -> {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
