#!/bin/bash
# Direct start for private-ai-server (avoid --reload, robust detach).
cd /mnt/d/ai-server-py
set -a; source .env 2>/dev/null; set +a
export HF_HOME=${HF_HOME:-/mnt/e/private-ai-server/models}
export GGUF_N_CTX=${GGUF_N_CTX:-4096}
exec .venv-wsl/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8005 --log-level info
