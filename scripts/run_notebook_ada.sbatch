#!/bin/bash
#SBATCH --job-name=proto_nb
#SBATCH --partition=ihub
#SBATCH --account=neuro
#SBATCH --qos=neuro
#SBATCH --nodelist=gnode103
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home2/narasimha.pai/rag_assignment/logs/%x_%j.out
# Executes notebooks/rag_baseline_vs_reranking.ipynb top to bottom on one RTX 3080 Ti and saves it with real outputs.
# Models are read from gnode103's node-local cache (offline), so nothing is downloaded.
set -e
source /home2/narasimha.pai/venv_llm/bin/activate
cd /home2/narasimha.pai/rag_assignment/repo/notebooks
export LLM_CACHE_DIR=/scratch/narasimha.pai/hf_cache
export BI_CACHE_DIR=/scratch/narasimha.pai/hf_cache
export CE_CACHE_DIR=/scratch/narasimha.pai/hf_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[$(date +%T)] $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import time, nbformat
from nbclient import NotebookClient
nb = nbformat.read("rag_baseline_vs_reranking.ipynb", as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "rag_baseline_vs_reranking.ipynb")
print(f"executed in {time.time() - t0:.0f}s")
PY
echo "[$(date +%T)] done"
