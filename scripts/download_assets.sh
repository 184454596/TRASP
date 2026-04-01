#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/Wangjl/TRASP"
RAW_ROOT="${PROJECT_ROOT}/data/raw"
EXTERNAL_ROOT="${PROJECT_ROOT}/external"
MODELS_ROOT="${PROJECT_ROOT}/models"
CACHE_ROOT="${PROJECT_ROOT}/.cache"
HF_HOME_DEFAULT="${CACHE_ROOT}/huggingface"
HF_DATASETS_CACHE_DEFAULT="${CACHE_ROOT}/datasets"

QWEN_MODEL_DIR="/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct"
GSM8K_DIR="${RAW_ROOT}/gsm8k"
XSTEST_DIR="${RAW_ROOT}/xstest"
ADVBENCH_DIR="${RAW_ROOT}/advbench"
HARMBENCH_REPO_DIR="${EXTERNAL_ROOT}/HarmBench"
HARMBENCH_BEHAVIOR_DIR="${HARMBENCH_REPO_DIR}/data/behavior_datasets"
HARMBENCH_CLS_DIR="${MODELS_ROOT}/HarmBench-Llama-2-13b-cls"

PRIMARY_HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
FALLBACK_HF_ENDPOINT="https://huggingface.co"
HF_DISABLE_PROXY="${HF_DISABLE_PROXY:-1}"

export HF_HOME="${HF_HOME:-${HF_HOME_DEFAULT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_DATASETS_CACHE_DEFAULT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

mkdir -p "${RAW_ROOT}" "${EXTERNAL_ROOT}" "${MODELS_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

log() {
  printf '[download_assets] %s\n' "$1"
}

run_hf_command() {
  if [[ "${HF_DISABLE_PROXY}" == "1" ]]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy "$@"
    return $?
  fi

  "$@"
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    printf '[download_assets] missing required directory for %s: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
  log "found ${label}: ${path}"
}

validate_saved_dataset() {
  local target_dir="$1"
  python - "${target_dir}" <<'PY'
from datasets import load_from_disk
from pathlib import Path
import sys

target = Path(sys.argv[1])
if not target.exists():
    raise SystemExit(1)

try:
    load_from_disk(str(target))
except Exception as exc:
    print(f"[download_assets] dataset validation failed for {target}: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
}

download_dataset() {
  local repo_id="$1"
  local config_name="$2"
  local target_dir="$3"
  local label="$4"

  if [[ -d "${target_dir}" ]]; then
    if validate_saved_dataset "${target_dir}"; then
      log "skip ${label}; existing dataset is loadable at ${target_dir}"
      return 0
    fi
    printf '[download_assets] existing directory for %s is not a valid saved dataset: %s\n' "${label}" "${target_dir}" >&2
    exit 1
  fi

  local endpoint
  for endpoint in "${PRIMARY_HF_ENDPOINT}" "${FALLBACK_HF_ENDPOINT}"; do
    log "downloading ${label} from ${repo_id} via ${endpoint}"
    if run_hf_command env HF_ENDPOINT="${endpoint}" python - "${repo_id}" "${config_name}" "${target_dir}" <<'PY'
from datasets import DatasetDict, load_dataset, load_dataset_builder
from pathlib import Path
import shutil
import sys

repo_id, config_name, target_dir = sys.argv[1:4]
kwargs = {}
if config_name != "__NONE__":
    kwargs["name"] = config_name

target = Path(target_dir)
tmp_target = target.with_name(target.name + ".tmp")
if tmp_target.exists():
    shutil.rmtree(tmp_target)

builder = load_dataset_builder(repo_id, **kwargs)
split_infos = builder.info.splits or {}
non_empty_splits = [
    split_name
    for split_name, split_info in split_infos.items()
    if getattr(split_info, "num_examples", None) not in (None, 0)
]

if split_infos and non_empty_splits and len(non_empty_splits) != len(split_infos):
    dataset = DatasetDict(
        {split_name: load_dataset(repo_id, split=split_name, **kwargs) for split_name in non_empty_splits}
    )
else:
    dataset = load_dataset(repo_id, **kwargs)

dataset.save_to_disk(str(tmp_target))
tmp_target.replace(target)
print(f"[download_assets] saved {repo_id} to {target}")
PY
    then
      validate_saved_dataset "${target_dir}"
      log "completed ${label}: ${target_dir}"
      return 0
    fi
    log "download attempt failed for ${label} via ${endpoint}"
  done

  printf '[download_assets] failed to download %s from %s\n' "${label}" "${repo_id}" >&2
  exit 1
}

validate_harmbench_classifier() {
  local target_dir="$1"
  [[ -f "${target_dir}/config.json" ]] || return 1
  [[ -f "${target_dir}/tokenizer_config.json" ]] || return 1

  if compgen -G "${target_dir}/.cache/huggingface/download/*.incomplete" > /dev/null; then
    return 1
  fi

  if [[ -f "${target_dir}/model.safetensors.index.json" ]]; then
    compgen -G "${target_dir}/model-*.safetensors" > /dev/null
    return $?
  fi

  compgen -G "${target_dir}/model.safetensors" > /dev/null
}

download_hf_model_snapshot() {
  local endpoint="$1"

  if command -v huggingface-cli > /dev/null 2>&1; then
    run_hf_command env HF_ENDPOINT="${endpoint}" huggingface-cli download \
      cais/HarmBench-Llama-2-13b-cls \
      --local-dir "${HARMBENCH_CLS_DIR}"
    return 0
  fi

  run_hf_command env HF_ENDPOINT="${endpoint}" python - "${HARMBENCH_CLS_DIR}" <<'PY'
from huggingface_hub import snapshot_download
import sys

target_dir = sys.argv[1]
snapshot_download(
    repo_id="cais/HarmBench-Llama-2-13b-cls",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
PY
}

download_harmbench_classifier() {
  if [[ -d "${HARMBENCH_CLS_DIR}" ]]; then
    if validate_harmbench_classifier "${HARMBENCH_CLS_DIR}"; then
      log "skip HarmBench classifier; existing files look complete at ${HARMBENCH_CLS_DIR}"
      return 0
    fi
    log "resuming incomplete HarmBench classifier download at ${HARMBENCH_CLS_DIR}"
  else
    mkdir -p "${HARMBENCH_CLS_DIR}"
  fi

  local endpoint
  for endpoint in "${PRIMARY_HF_ENDPOINT}" "${FALLBACK_HF_ENDPOINT}"; do
    log "downloading HarmBench classifier via ${endpoint}"
    if download_hf_model_snapshot "${endpoint}"; then
      if validate_harmbench_classifier "${HARMBENCH_CLS_DIR}"; then
        log "completed HarmBench classifier: ${HARMBENCH_CLS_DIR}"
        return 0
      fi
    fi
    log "download attempt failed for HarmBench classifier via ${endpoint}"
  done

  printf '[download_assets] failed to download HarmBench classifier\n' >&2
  exit 1
}

clone_harmbench_repo() {
  if [[ -d "${HARMBENCH_REPO_DIR}/.git" && -d "${HARMBENCH_BEHAVIOR_DIR}" ]]; then
    log "skip HarmBench repo; behavior datasets already present at ${HARMBENCH_REPO_DIR}"
    return 0
  fi

  if [[ -e "${HARMBENCH_REPO_DIR}" && ! -d "${HARMBENCH_REPO_DIR}/.git" ]]; then
    printf '[download_assets] target path exists but is not a git checkout: %s\n' "${HARMBENCH_REPO_DIR}" >&2
    exit 1
  fi

  if [[ -d "${HARMBENCH_REPO_DIR}/.git" ]]; then
    log "existing HarmBench checkout detected; verifying contents"
  else
    log "cloning HarmBench official repository"
    git clone --depth 1 https://github.com/centerforaisafety/HarmBench.git "${HARMBENCH_REPO_DIR}"
  fi

  if [[ ! -d "${HARMBENCH_BEHAVIOR_DIR}" ]]; then
    printf '[download_assets] HarmBench clone is missing behavior datasets directory: %s\n' "${HARMBENCH_BEHAVIOR_DIR}" >&2
    exit 1
  fi

  log "completed HarmBench repo: ${HARMBENCH_REPO_DIR}"
}

main() {
  log "project root: ${PROJECT_ROOT}"
  log "HF endpoint priority: ${PRIMARY_HF_ENDPOINT} -> ${FALLBACK_HF_ENDPOINT}"
  log "HF proxy disabled: ${HF_DISABLE_PROXY}"
  log "HF cache: ${HF_HOME}"
  log "datasets cache: ${HF_DATASETS_CACHE}"

  require_dir "${QWEN_MODEL_DIR}" "local target model"
  log "target model will be reused in-place and will not be downloaded: ${QWEN_MODEL_DIR}"

  download_dataset "openai/gsm8k" "main" "${GSM8K_DIR}" "GSM8K"
  download_dataset "Paul/XSTest" "__NONE__" "${XSTEST_DIR}" "XSTest"
  download_dataset "AlignmentResearch/AdvBench" "default" "${ADVBENCH_DIR}" "AdvBench"
  clone_harmbench_repo
  download_harmbench_classifier

  log "all requested assets are ready"
  log "run verification with: python /home/Wangjl/TRASP/scripts/verify_assets.py"
}

main "$@"
