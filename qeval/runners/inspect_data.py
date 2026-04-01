from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qeval.common.config import get_manifest_paths, get_required_fields, load_protocol_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect built manifest files for the base Qwen eval protocol.')
    parser.add_argument('--config', required=True, help='Absolute path to the protocol YAML.')
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def inspect_manifest(path: Path, required_fields: list[str]) -> tuple[int, list[dict], list[str]]:
    count = 0
    preview: list[dict] = []
    errors: list[str] = []

    for line_number, record in iter_jsonl(path):
        count += 1
        missing = [field for field in required_fields if field not in record]
        if missing:
            errors.append(f'{path.name}: line {line_number} missing fields {missing}')
        if len(preview) < 3:
            preview.append(record)

    return count, preview, errors


def main() -> int:
    args = parse_args()
    bundle = load_protocol_bundle(args.config)
    required_fields = get_required_fields(bundle)
    manifest_paths = get_manifest_paths(bundle)

    all_errors: list[str] = []
    for manifest_name, manifest_path in manifest_paths.items():
        print(f'== {manifest_name} ==')
        print(f'path: {manifest_path}')
        if not manifest_path.is_file():
            print('status: MISSING')
            all_errors.append(f'{manifest_path} does not exist')
            continue

        count, preview, errors = inspect_manifest(manifest_path, required_fields)
        print(f'count: {count}')
        print(f'required_fields: {required_fields}')
        print('samples:')
        for sample in preview:
            print(json.dumps(sample, ensure_ascii=False, sort_keys=True))
        if errors:
            print('field_validation: FAILED')
            for error in errors[:10]:
                print(error)
        else:
            print('field_validation: OK')
        print()
        all_errors.extend(errors)

    return 1 if all_errors else 0


if __name__ == '__main__':
    sys.exit(main())
