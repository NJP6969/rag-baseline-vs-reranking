# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked copy of the **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1)**:

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Candidates | top 10 by cosine similarity from a FAISS index of `all-MiniLM-L6-v2` embeddings | the same top 10 |
| Ordering | bi-encoder score | `cross-encoder/ms-marco-MiniLM-L-6-v2` score |
| LLM context | top 3 | top 3 after re-ranking |
| Generator | `Qwen/Qwen2.5-14B-Instruct`, 4-bit NF4, greedy, identical prompt | same |

## Repository layout

```
notebooks/rag_baseline_vs_reranking.ipynb   main deliverable, executed end to end on a GPU (all outputs are real)
data/nist_ai_rmf.pdf                        the source document
data/results.json                           answers, rank shifts and latencies written by the notebook
data/failure_query_search.json              output of the failure-query search (33 candidate queries)
scripts/find_failure_queries.py             how the stress-test queries were found
scripts/run_notebook_ada.sbatch             SLURM job used to execute the notebook on the IIIT Ada cluster
```

## How the stress-test queries were chosen

`scripts/find_failure_queries.py` pairs each candidate query with a phrase taken from the passage that actually answers it. It locates that answer chunk and records its rank under the bi-encoder (over all chunks) and under the cross-encoder (within the top 10). Thirty-three candidates were written across the assignment's categories: keyword trap, exceptions, negation, and buried chunks. The notebook uses the five where the answer chunk was at rank 4–10 and the cross-encoder moved it into the top 3, plus one case where stage 1 never retrieves the answer at all.

## Running it

Requirements: Python 3.10, a CUDA GPU with at least 12 GB of memory (for the 14B model in 4-bit), and

```bash
pip install torch transformers sentence-transformers bitsandbytes accelerate faiss-cpu pypdf pandas jupyter
```

Then open `notebooks/rag_baseline_vs_reranking.ipynb` and run all cells, from inside `notebooks/`. The models download from Hugging Face on first use. To use existing local caches instead, set `LLM_CACHE_DIR`, `BI_CACHE_DIR` and `CE_CACHE_DIR`.

On Ada the notebook was executed non-interactively with `sbatch scripts/run_notebook_ada.sbatch` on one RTX 3080 Ti (12 GB).
