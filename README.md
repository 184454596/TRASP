# TRASP Base Model Evaluation Scaffold

This repository is currently scoped to a minimal, auditable evaluation project for the base model `Qwen2.5-7B-Instruct`.

Current project goal:
- Evaluate the base aligned model only.
- Target local model: `/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct`
- Do not implement TRASP training.
- Do not implement other baselines.
- Do not implement general training pipelines.

The current stage only initializes the environment and project skeleton. Training logic is intentionally excluded.

## Project Layout

```text
/home/Wangjl/TRASP
├── configs
├── data
│   ├── raw
│   └── processed
├── external
├── outputs
├── qeval
│   ├── common
│   ├── data
│   ├── eval
│   ├── models
│   ├── runners
│   └── transforms
├── scripts
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Activate the environment

The expected conda environment is `TRASP`.

```bash
conda activate TRASP
```

### 2. Install dependencies

Install dependencies into the currently activated conda environment:

```bash
bash /home/Wangjl/TRASP/scripts/install_deps.sh
```

This script will:
- install packages from `/home/Wangjl/TRASP/requirements.txt`
- download the spaCy English model `en_core_web_sm`
- print the installed `python`, `torch`, and `transformers` versions

### 3. Confirm the local target model path

The project assumes the target model already exists locally and should not be downloaded again:

```text
/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct
```

### 4. Next steps in this repository

The next implementation stages will add:
- dataset and judge download/bootstrap scripts under `scripts/`
- YAML protocol configs under `configs/`
- evaluation runners under `qeval/runners/`
- machine-readable outputs under `outputs/`

Planned entry points:
- data/bootstrap: `scripts/bootstrap_assets.py`
- clean evaluation: `scripts/run_clean_eval.py`
- transform-stress evaluation: `scripts/run_transform_eval.py`
- final aggregation: `scripts/aggregate_metrics.py`

These entry points are documented here in advance, but they are not implemented in this initialization stage.

## Asset Download

Public assets in this stage do not require a Hugging Face account. The download script uses a Hugging Face mirror by default and falls back to the official endpoint if needed.
By default, the script disables `HTTP_PROXY` and `HTTPS_PROXY` for Hugging Face downloads because a broken local proxy can interrupt large model transfers.

### 1. Download datasets and judge assets

```bash
bash /home/Wangjl/TRASP/scripts/download_assets.sh
```

This script will:
- reuse the local target model at `/home/Wangjl/models/Qwen/Qwen2.5-7B-Instruct`
- download GSM8K to `/home/Wangjl/TRASP/data/raw/gsm8k`
- download XSTest to `/home/Wangjl/TRASP/data/raw/xstest`
- download AdvBench to `/home/Wangjl/TRASP/data/raw/advbench`
- clone the official HarmBench repository to `/home/Wangjl/TRASP/external/HarmBench`
- download the official HarmBench classifier to `/home/Wangjl/TRASP/models/HarmBench-Llama-2-13b-cls`

The script is idempotent:
- existing valid dataset directories are skipped
- an existing valid HarmBench classifier directory is skipped
- the local Qwen target model is only checked and never re-downloaded

### 2. Verify assets

```bash
python /home/Wangjl/TRASP/scripts/verify_assets.py
```

The verification script checks:
- the local Qwen model directory
- the HarmBench classifier directory
- the local GSM8K / XSTest / AdvBench directories
- the HarmBench repository checkout
- the `behavior_datasets` directory inside HarmBench

### 3. Retry guidance

If a Hugging Face dataset or model download fails:
- rerun `bash /home/Wangjl/TRASP/scripts/download_assets.sh`
- the script will retry against the mirror first, then the official Hugging Face endpoint
- partial completed directories are not silently overwritten; if a directory is incomplete, fix or remove it before retrying

You can also override the mirror endpoint manually:

```bash
HF_ENDPOINT=https://hf-mirror.com bash /home/Wangjl/TRASP/scripts/download_assets.sh
```

If the mirror is unavailable, you can force the official endpoint:

```bash
HF_ENDPOINT=https://huggingface.co bash /home/Wangjl/TRASP/scripts/download_assets.sh
```

If you intentionally need to keep your current proxy settings for Hugging Face downloads:

```bash
HF_DISABLE_PROXY=0 bash /home/Wangjl/TRASP/scripts/download_assets.sh
```
