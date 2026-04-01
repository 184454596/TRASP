from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_protocol_bundle(protocol_path: str | Path) -> dict[str, Any]:
    protocol = load_yaml(protocol_path)
    bundle: dict[str, Any] = {
        'protocol_path': str(Path(protocol_path).expanduser().resolve()),
        'protocol': protocol,
    }
    for name, include_path in protocol.get('includes', {}).items():
        bundle[name] = load_yaml(include_path)
    return bundle


def get_required_fields(bundle: dict[str, Any]) -> list[str]:
    return list(bundle['datasets']['schema']['required_fields'])


def get_manifests_dir(bundle: dict[str, Any]) -> Path:
    return Path(bundle['protocol']['paths']['manifests_dir']).expanduser().resolve()


def get_manifest_summary_path(bundle: dict[str, Any]) -> Path:
    return Path(bundle['protocol']['paths']['manifest_summary_path']).expanduser().resolve()


def get_output_root(bundle: dict[str, Any]) -> Path:
    return Path(bundle['protocol']['paths']['output_root']).expanduser().resolve()


def build_resolved_config(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        'protocol_path': bundle['protocol_path'],
        'protocol': bundle['protocol'],
        'datasets': bundle['datasets'],
        'judges': bundle['judges'],
        'metrics': bundle['metrics'],
        'transforms': bundle['transforms'],
    }


def get_manifest_paths(bundle: dict[str, Any]) -> OrderedDict[str, Path]:
    datasets_cfg = bundle['datasets']
    manifests_dir = get_manifests_dir(bundle)
    paths: OrderedDict[str, Path] = OrderedDict()

    names = [
        datasets_cfg['raw_sources']['harmbench_text']['output_manifest'],
        datasets_cfg['raw_sources']['advbench']['output_manifest'],
        datasets_cfg['derived_manifests']['harmful_eval_unified']['output_manifest'],
        datasets_cfg['raw_sources']['gsm8k']['output_manifests']['train'],
        datasets_cfg['raw_sources']['gsm8k']['output_manifests']['test'],
        datasets_cfg['raw_sources']['xstest']['output_manifest'],
    ]

    for name in names:
        paths[Path(name).stem] = manifests_dir / name

    return paths
