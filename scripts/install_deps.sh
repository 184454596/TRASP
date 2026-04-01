#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/Wangjl/TRASP"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "Using pip index: ${PIP_INDEX_URL}"

pip install -i "${PIP_INDEX_URL}" -r "${PROJECT_ROOT}/requirements.txt"
python -m spacy download en_core_web_sm

python - <<'PY'
import platform

import torch
import transformers

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
try:
    import google.protobuf
    print(f"protobuf={google.protobuf.__version__}")
except Exception:
    print("protobuf=NOT_INSTALLED")
PY
