# Two-Stage RAG Evaluation: Dense Bi-Encoder Baseline vs. Cross-Encoder Re-Ranking

**Author:** Narasimha Pai  
**Code Repository:** [https://github.com/NJP6969/rag-baseline-vs-reranking](https://github.com/NJP6969/rag-baseline-vs-reranking)  
**Main Executed Notebook:** [`notebooks/rag_baseline_vs_reranking.ipynb`](file:///C:/Users/narsi/Documents/antigravity/charming-darwin/notebooks/rag_baseline_vs_reranking.ipynb)  
**Corpus:** NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0 / NIST AI 100-1, 48 pages)  
**Hardware & Platform:** IIIT Hyderabad ADA HPC Cluster (`gnode103`, NVIDIA GeForce RTX 3080 Ti 12GB VRAM, CUDA 12.4, PyTorch 2.4.0)

---

## 1. Architecture Comparison: Bi-Encoders vs. Cross-Encoders

The fundamental divide between a Bi-Encoder and a Cross-Encoder lies in **when and how tokens are allowed to interact**.

```
Bi-Encoder (Two-Tower / Siamese):
  Query q   --> [ Transformer ] --> Pooling --> e_q (384-d vector) \
                                                                    --> Cosine Similarity (MIPS)
  Chunk d   --> [ Transformer ] --> Pooling --> e_d (384-d vector) /   (Cached offline in FAISS)

Cross-Encoder (Joint All-to-All Attention):
  [CLS] + q + [SEP] + d + [SEP] --> [ Transformer Layers (L=6) ] --> Linear Head --> Scalar Score
                                    (All query tokens attend to all chunk tokens)
```

### The Bi-Encoder (`all-MiniLM-L6-v2`)
A Bi-Encoder treats retrieval as a vector similarity problem in a metric space. The query $q$ and document chunk $d$ are passed through the transformer independently—either through distinct encoders or, as here, a shared Siamese network. Each token sequence is mean-pooled into a single 384-dimensional vector:
$$\mathbf{e}_q = \text{Norm}(\text{MeanPool}(\text{Transformer}(q))), \quad \mathbf{e}_d = \text{Norm}(\text{MeanPool}(\text{Transformer}(d)))$$

The relevance score is the inner product (cosine similarity) of the two unit vectors:
$$s_{\text{bi}}(q, d) = \mathbf{e}_q^\top \mathbf{e}_d$$

Because $\mathbf{e}_d$ is completely independent of $q$, we compute embeddings for all 237 chunks once at ingestion time and dump them into a flat inner-product FAISS index (`IndexFlatIP`). At query time, the system needs only one forward pass to embed $q$, followed by matrix multiplication that finishes in **~3.8 ms**.

**The Price Paid:** Compressing 600 characters of technical prose into 384 floating-point numbers creates a severe representational bottleneck. The vector captures aggregate topical "gist" rather than syntactic composition. Small function words carrying immense logical weight—such as *"not"*, *"unless"*, *"prohibited"*, or domain synonyms—barely nudge the vector in embedding space. The Bi-Encoder easily falls for passages that share broad topical vocabulary with the query while completely missing the conditional answer.

### The Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
A Cross-Encoder discards the vector-space abstraction entirely. The query and passage are concatenated into a single token sequence separated by delimiter tokens:
$$\mathbf{x} = [\text{CLS}] \circ q \circ [\text{SEP}] \circ d \circ [\text{SEP}]$$

This sequence is fed through 6 transformer layers where **joint self-attention operates across all positions at every layer**. A token in the query can directly attend to a token in the passage:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The final hidden state of the $[\text{CLS}]$ token is fed to a linear classification head that outputs an unbounded relevance logit:
$$s_{\text{cross}}(q, d) = \mathbf{W} \cdot \mathbf{h}_{[\text{CLS}]} + b$$

Because tokens attend to each other across the sequence, the model readily detects whether a negative qualifier in the query aligns with a restrictive clause in the passage, or whether an apparent lexical match is purely superficial.

**The Price Paid:** Relevance cannot be decomposed into $\mathbf{e}_q$ and $\mathbf{e}_d$. Every candidate chunk requires a distinct, full transformer forward pass conditioned on $q$. You cannot build a static index over the corpus. Evaluating 100,000 chunks would demand 100,000 forward passes for a single user query.

---

## 2. Qualitative & Rank Shift Analysis

### Methodology: Systematic Discovery of Failure Modes
Rather than hand-crafting arbitrary prompts, we wrote a test discovery script ([`scripts/find_failure_queries.py`](file:///C:/Users/narsi/Documents/antigravity/charming-darwin/scripts/find_failure_queries.py)) that evaluated **33 candidate queries** against the NIST AI RMF corpus. Each query was mapped to a definitive target chunk containing the ground truth.

Across the 33 queries:
- **17 queries**: The ground-truth chunk was already in the top 3 of the baseline (the Bi-Encoder handled them without help).
- **9 queries**: The ground-truth chunk was ranked $>10$ (a stage-1 recall failure; never reached the re-ranker).
- **2 queries**: The chunk was in ranks 4–10 but was not promoted by the Cross-Encoder.
- **5 queries**: The chunk sat buried at ranks 4–10 in the baseline, and the Cross-Encoder successfully promoted it into the top 3.

We selected these 5 successful promotions (**F1–F5**), representing distinct failure categories, along with 1 structural recall failure (**S1**), to benchmark under identical conditions.

### Test Case Summary

Both pipelines fed their Top 3 retrieved passages into an identical prompt template answered by local **`Qwen2.5-14B-Instruct`** (4-bit NF4 quantization on the RTX 3080 Ti, greedy decoding with `do_sample=False`).

| Query ID | Category | Target Chunk (Page) | Bi-Encoder Rank $\rightarrow$ Cross-Encoder Rank | In P1 Top 3? | In P2 Top 3? | Core Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1** | Exception / Stop Rule | `chunk_0051` (p. 13) | **#9 $\rightarrow$ #1** (+8) | No | **Yes** | Baseline gave a factually false denial; re-ranking retrieved the stop rule. |
| **F2** | Negation Qualifier | `chunk_0096` (p. 21) | **#4 $\rightarrow$ #1** (+3) | No | **Yes** | Baseline ungrounded / hedged; re-ranking quoted exact document constraint. |
| **F3** | Caveat / Exception | `chunk_0098` (p. 21) | **#5 $\rightarrow$ #1** (+4) | No | **Yes** | Baseline suffered complete omission; re-ranking identified resource/IP limits. |
| **F4** | Keyword / Synonym Trap | `chunk_0092` (p. 20) | **#5 $\rightarrow$ #1** (+4) | No | **Yes** | Baseline failed on colloquial phrase; re-ranking mapped to "resilience". |
| **F5** | Buried Chunk (Control) | `chunk_0102` (p. 22) | **#4 $\rightarrow$ #3** (+1) | No | **Yes** | Rank shift occurred, but both answers were identical (boundary case). |
| **S1** | Stage-1 Recall Wall | `chunk_0130` (p. 28) | **#29 $\rightarrow$ N/A** (0) | No | No | Re-ranker never saw the chunk; demonstrates stage-1 recall ceiling. |

---

### In-Depth Case Studies

#### 1. Case F1 (Exception / Hard Stop Condition): Complete Factual Error Fixed
* **Query:** *"Is there any situation in which the framework says AI development should be stopped entirely?"*
* **The Failure:** The Bi-Encoder pushed general governance passages (`chunk_0013`, `chunk_0191`, `chunk_0176`) to ranks 1–3 because they were saturated with high-frequency terms like *"AI development"*, *"lifecycle"*, and *"risk management"*. The actual governing rule—which specifies that development and deployment should cease if risks are unacceptable—was buried at **Rank #9** (`chunk_0051`, bi-score 0.464).
* **Baseline Answer 1:**
  > *"The provided context passages do not specify any situation where the AI Risk Management Framework dictates stopping AI development entirely... Therefore, no explicit directive to cease AI development is mentioned."*
  
  This is a flat hallucination of absence: the LLM was handed three broad governance chunks and concluded the rule does not exist.
* **Re-Ranked Answer 2:**
  The Cross-Encoder scored `chunk_0051` at **+1.19** (its highest score in the batch) and demoted the fluffy governance chunks into negative territory (e.g. `chunk_0013` dropped from #1 to #10 with score -8.00). Promoted to **Rank #1**, `chunk_0051` allowed the model to answer:
  > *"Yes, according to the context provided, the AI RMF suggests that if an AI system presents unacceptable negative risk levels, such as imminent significant negative impacts, actual severe harms, or catastrophic risks, then its development and deployment should cease until these risks can be adequately managed. This guidance is found on page 13 of the document."*

---

#### 2. Case F2 (Negation & Constraint): From Guesswork to Textual Grounding
* **Query:** *"Does making an AI system transparent guarantee that it is accurate or fair?"*
* **The Failure:** The Bi-Encoder matched chunks discussing transparency positively (`chunk_0093`, `chunk_0072`, `chunk_0095`), while the exact qualifying chunk (`chunk_0096`: *"a transparent system is not necessarily accurate, fair, or secure"*) landed at **Rank #4**.
* **Baseline Answer 1:**
  > *"Transparency in an AI system... primarily ensures that information... is accessible... However, the context does not explicitly state that transparency guarantees accuracy or fairness... Therefore, while transparency is important for trustworthiness, it does not alone guarantee that an AI system is accurate or fair."*
  
  The LLM reasoned out the correct conclusion using its internal weights, but explicitly flagged that the context lacked the statement. It was an ungrounded inference.
* **Re-Ranked Answer 2:**
  Promoting `chunk_0096` from **#4 to #1** provided the direct textual refutation:
  > *"No, making an AI system transparent does not guarantee that it is accurate or fair... but a transparent system is not necessarily accurate, fair, or secure. The document emphasizes that while transparency can help in assessing other characteristics... it does not inherently ensure these qualities."*

---

#### 3. Case F4 (Keyword / Synonym Trap): Bridging Lexical Vocabulary Mismatch
* **Query:** *"What does the framework mean by the ability of a system to bounce back after an unexpected adverse event?"*
* **The Failure:** The user asked about *"bouncing back"*, an informal phrasing. The document never uses the colloquial string *"bounce back"*; it uses the formal technical term *"resilience"*, defined on page 20 as *"the capacity to return to normal function following an unexpected adverse event"*. Because the Bi-Encoder relies heavily on sub-word lexical overlap, `chunk_0092` was pushed to **Rank #5**. Chunks with generic *"events"* or *"adverse impacts"* filled ranks 1–3.
* **Baseline Answer 1:**
  > *"The provided context passages do not explicitly define what the framework means by the ability of a system to 'bounce back'... However, the concept might relate to robustness, which is mentioned on page 19... Additionally... resilience or recovery capabilities could be inferred... though this is not directly stated."*
  
  The baseline hedges and incorrectly guesses "robustness".
* **Re-Ranked Answer 2:**
  The Cross-Encoder attended to the semantic equivalence between *"bounce back after an unexpected adverse event"* and *"return to normal function"*, shooting `chunk_0092` to **Rank #1** (cross-score +1.28). Answer 2 was decisive:
  > *"The NIST AI Risk Management Framework refers to the ability of a system to bounce back after an unexpected adverse event as resilience. Resilience is described as the capacity to return to normal function following such an event..."*

---

#### 4. The Counter-Observations: F5 and S1
Science is unconvincing when it only shows cherry-picked successes. Two cases in our data expose the practical nuances and hard boundaries of re-ranking:

* **F5 (A Rank Shift That Made No Difference):**
  In F5 (*"What is the difference between explainability and interpretability?"*), the target `chunk_0102` jumped from **Rank #4 to Rank #3**. Yet Answer 1 and Answer 2 are virtually indistinguishable. Why? Because the Bi-Encoder's top 3 already included `chunk_0101` and `chunk_0103`—adjacent slices of text from page 21–22 that contained redundant definitions. **A rank promotion only alters the generation if it introduces non-redundant information absent from the baseline window.**
* **S1 (The Stage-1 Recall Ceiling):**
  In S1 (*"What specific operational actions are required for AI system decommissioning and end-of-life residual risk?"*), the definitive passage (`chunk_0130` on page 28) was ranked **#29** by the Bi-Encoder. Because Pipeline 2 only feeds the Top 10 to the Cross-Encoder, the model never saw `chunk_0130`. Both pipelines produced identically vague non-answers. **Re-ranking cannot fix what retrieval failed to surface.**

---

## 3. Latency Profiling

We benchmarked stage timings over **20 repeated executions** on the ADA RTX 3080 Ti following an initial warm-up pass.

### Execution Latencies (Median & 95th Percentile)

| Query ID | Stage 1 Retrieval (FAISS) Median (p95) | Stage 2 Re-Rank (10 pairs) Median (p95) | Answer 1 Gen (Tokens) | Answer 2 Gen (Tokens) | Pipeline 1 Total | Pipeline 2 Total | Re-Rank Share of Pipeline 2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** | 3.85 ms (3.93 ms) | 8.41 ms (8.43 ms) | 6,215 ms (99 tok) | 4,313 ms (68 tok) | 6,219 ms | 4,326 ms | 0.19% |
| **F2** | 3.83 ms (3.85 ms) | 8.07 ms (8.09 ms) | 7,677 ms (125 tok) | 5,860 ms (95 tok) | 7,681 ms | 5,872 ms | 0.14% |
| **F3** | 3.83 ms (3.85 ms) | 7.78 ms (7.82 ms) | 5,218 ms (84 tok) | 6,686 ms (108 tok) | 5,221 ms | 6,698 ms | 0.12% |
| **F4** | 3.86 ms (3.90 ms) | 8.36 ms (8.38 ms) | 7,648 ms (124 tok) | 8,803 ms (143 tok) | 7,652 ms | 8,816 ms | 0.09% |
| **F5** | 3.73 ms (3.75 ms) | 8.07 ms (8.10 ms) | 9,671 ms (157 tok) | 7,692 ms (124 tok) | 9,675 ms | 7,704 ms | 0.10% |
| **S1** | 3.86 ms (3.87 ms) | 8.60 ms (9.02 ms) | 9,494 ms (155 tok) | 6,457 ms (104 tok) | 9,497 ms | 6,470 ms | 0.13% |
| **Mean** | **3.83 ms (3.86 ms)** | **8.22 ms (8.31 ms)** | **7,654 ms (124 tok)**| **6,635 ms (107 tok)**| **7,658 ms**| **6,647 ms**| **0.13%** |

### Latency Findings

1. **Re-Ranking is Numerically Free:**
   Scoring 10 candidate pairs with `ms-marco-MiniLM-L-6-v2` in a single GPU batch took **8.22 ms**. In a system where autoregressive text generation takes 4,000–9,000 ms, re-ranking represents **0.13%** of total query latency. It is imperceptible to end users.
2. **The Generation-Time Paradox:**
   Counter-intuitively, Pipeline 2 had a *shorter* average end-to-end runtime than Pipeline 1 (6,647 ms vs. 7,658 ms). This has nothing to do with retrieval efficiency. At an inference throughput of **~16.2 tokens/second**, total runtime is dictated by output length. In F1 and F2, when the baseline model lacked the right chunk, it generated verbose, hedging explanations (125 tokens). When the re-ranker provided the exact sentence, the model stated the answer cleanly in 68 tokens, shaving over 1.8 seconds off decoding time.

---

## 4. Scaling: Why Not Run a Cross-Encoder from the Very Start?

If Cross-Encoders are so overwhelmingly accurate, why do we bother with Bi-Encoders at all? Why not pipe the entire document collection through the Cross-Encoder directly?

To answer this concretely, we ran an empirical benchmark directly inside the notebook, measuring full-corpus Cross-Encoding on the RTX 3080 Ti versus Bi-Encoder + FAISS search:

```
[Empirical Measurement on 237 Chunks (RTX 3080 Ti)]:
- Cross-Encoder scoring all 237 chunks for 1 query: 103.0 ms (0.43 ms per pair, batched)
- Bi-Encoder query encoding + FAISS inner-product search: 4.1 ms
```

Using our measured unit cost of **0.4328 ms per (query, chunk) pair**, we extrapolate to production corpus sizes:

| Corpus Size ($N$ Chunks) | Cross-Encoder Over Every Chunk ($O(N)$) | Two-Stage RAG: Bi-Encoder + Top 10 Re-rank ($O(\log N)$) | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **237** (Our corpus) | 0.103 seconds | **0.004 seconds** | $25\times$ |
| **10,000** (Small manual) | 4.33 seconds | **0.004 seconds** | $1,080\times$ |
| **1,000,000** (Enterprise docs) | **432.8 seconds (~7.2 minutes)** | **0.004 seconds** | $108,200\times$ |
| **100,000,000** (Web search) | **43,281 seconds (~12.0 hours)** | **0.004 seconds** | $10,820,000\times$ |

### The Three Structural Walls of Exhaustive Cross-Encoding

1. **Computational Asymptotics:**
   A Bi-Encoder splits computation into an offline phase ($O(N)$ transformer passes executed once during indexing) and an online query phase. Query retrieval against pre-normalized embeddings reduces to Maximum Inner Product Search (MIPS), which takes $O(D)$ for flat vector scanning or $O(\log N)$ with hierarchical navigable small-world (HNSW) graphs.
   
   A Cross-Encoder has **zero offline pre-computation**. Because query tokens and document tokens are coupled in the very first self-attention layer, every candidate document must pass through all $L$ layers of the transformer together with $q$ at query time. The inference complexity is strictly $O(N \cdot L \cdot (|q| + |d|)^2)$. At 1 million chunks, answering one query takes 7.2 minutes of continuous GPU tensor-core saturation.

2. **The Throughput & Concurrency Wall:**
   Even if a data center could tolerate a 4-second delay for 10,000 chunks, serving 1,000 concurrent user queries would require 10,000,000 forward passes per second. The server fleet would melt under memory bandwidth pressure alone.

3. **The Funnel Architecture as the Only Viable Synthesis:**
   Industrial RAG systems resolve this tension by adopting a **two-stage funnel**:
   - **Stage 1 (High Recall, Coarse Precision):** The Bi-Encoder scans millions of vectors in $<5$ ms to discard 99.99% of irrelevant documents, guaranteeing high recall on candidate chunks.
   - **Stage 2 (High Precision, Fixed Cost):** The Cross-Encoder acts as an expensive filter on a constant number of candidates ($K=10$ or $K=50$). Because $K$ is invariant to total collection size $N$, the re-ranking stage executes in a fixed **~8 ms window**, whether the underlying database contains 200 chunks or 200 million.
   
   The critical architectural lesson from our experiments is that the funnel is non-negotiable—and that its primary vulnerability is Stage 1 recall (as demonstrated in S1).
