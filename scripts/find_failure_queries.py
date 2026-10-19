"""Search for genuine failure queries: answer chunk at bi-encoder rank 4-10 that the cross-encoder promotes into the top 3.

Usage:
  python scripts/find_failure_queries.py data/nist_ai_rmf.pdf --check         # only verify every target phrase occurs in some chunk (no models)
  python scripts/find_failure_queries.py data/nist_ai_rmf.pdf data/failure_query_search.json   # full run (bi + cross encoder)
"""
import json, re, sys
from pypdf import PdfReader


def chunk_pdf(pdf, chunk_size=600, overlap=120):
    chunks = []
    for pn, page in enumerate(PdfReader(pdf).pages, 1):
        raw = page.extract_text()
        if not raw:
            continue
        c = " ".join(raw.split())
        if len(c) < 50:
            continue
        s = 0
        while s < len(c):
            e = min(s + chunk_size, len(c))
            if e < len(c):
                ls = c.rfind(" ", s, e)
                if ls > s + chunk_size // 2:
                    e = ls
            t = c[s:e].strip()
            if len(t) >= 80:
                chunks.append({"chunk_id": f"chunk_{len(chunks):04d}", "page": pn, "text": t})
            if e >= len(c):
                break
            s = e - overlap
    return chunks


def norm(s):
    # undo pypdf line-break hyphenation ("devel- opers") and smart quotes so phrases match
    s = re.sub(r"(\w)- (\w)", r"\1\2", s)
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


# (category, query, target phrase that only the answering passage contains)
CANDIDATES = [
    # --- negation & constraints
    ("negation", "If an AI risk cannot be measured, does that mean the system is low risk?",
     "inability to appropriately measure ai risks does not imply"),
    ("negation", "Does making an AI system transparent guarantee that it is accurate or fair?",
     "a transparent system is not necessarily an accurate"),
    ("negation", "Should organizations try to eliminate all negative AI risk?",
     "attempting to eliminate negative risk entirely can be counterproductive"),
    ("negation", "Does mitigating harmful bias automatically make an AI system fair?",
     "systems in which harmful biases are mitigated are not necessarily fair"),
    ("negation", "Is following the AI RMF legally mandatory for companies?",
     "voluntary"),
    ("negation", "What level of risk tolerance does the framework require organizations to adopt?",
     "it does not prescribe risk tolerance"),
    ("negation", "Which AI systems are prohibited from being deployed under the framework?",
     "development and deployment should cease in a safe manner"),
    ("negation", "Is the AI RMF meant to be applied as a one-size-fits-all checklist?",
     "checklist"),
    # --- exceptions & corner cases ("except when", conditions, stop rules)
    ("exception", "Is there any situation in which the framework says AI development should be stopped entirely?",
     "development and deployment should cease in a safe manner"),
    ("exception", "Under what conditions can privacy-enhancing techniques reduce model accuracy?",
     "under certain conditions such as data sparsity"),
    ("exception", "When should AI developers adjust their transparency and accountability practices?",
     "when consequences are severe such as when life and liberty are at stake"),
    ("exception", "What limits are placed on transparency because of the cost to the organization?",
     "need to safeguard proprietary information"),
    ("exception", "When can risk measured in a laboratory differ from risk in the real world?",
     "measuring ai risks in a laboratory or a controlled environment may"),
    ("exception", "Can a risk measured early in the AI lifecycle change later on?",
     "measuring risk at an earlier stage in the ai lifecycle may yield different results"),
    ("exception", "Why can third-party software and data make AI risk harder to measure?",
     "thirdparty software hardware and data"),
    ("exception", "What should happen when an AI system shows outcomes inconsistent with its intended use?",
     "supersede disengage or deactivate ai systems"),
    # --- keyword traps / synonyms
    ("keyword_trap", "How is resilience different from security for an AI system?",
     "can maintain confidentiality integrity and availability through protection mechanisms"),
    ("keyword_trap", "What does the framework mean by the ability of a system to bounce back after an unexpected adverse event?",
     "return to normal function after an unexpected adverse event"),
    ("keyword_trap", "What is the difference between explainability and interpretability?",
     "explainability can answer the question of how"),
    ("keyword_trap", "What does validation mean in the AI RMF?",
     "confirmation through the provision of objective evidence"),
    ("keyword_trap", "Who are the people that should be consulted outside the organization building the AI?",
     "affected communities"),
    ("keyword_trap", "How should an organization retire an AI system safely?",
     "decommissioning and phasing out ai systems safely"),
    ("keyword_trap", "What does the framework say about humans overseeing AI decisions and their own biases?",
     "human cognitive biases"),
    ("keyword_trap", "What are the three categories of AI bias?",
     "systemic computational and statistical and humancognitive"),
    ("keyword_trap", "What is a current profile versus a target profile?",
     "current profile indicates how ai is currently being managed"),
    # --- other buried-chunk / document-specific
    ("other", "What does the Measure function do?",
     "measure function employs quantitative qualitative or mixedmethod tools"),
    ("other", "Why is it hard to compare AI risk with risks from human baselines?",
     "human baseline"),
    ("other", "Why are pre-trained models a source of risk?",
     "pretrained models"),
    ("other", "Who should perform test, evaluation, verification and validation tasks?",
     "tevv"),
    ("other", "What does the framework say about how frequently risk management should be reviewed?",
     "determining the frequency of periodic review"),
    # --- original report queries (kept for comparison)
    ("orig_q1", "What specific operational actions are required for AI system decommissioning and end-of-life residual risk?",
     "decommissioning and phasing out ai systems safely"),
    ("orig_q2", "Under what circumstances is transparency intentionally restricted or not advisable according to the framework?",
     "need to safeguard proprietary information"),
    ("orig_q3", "How can increasing model explainability directly undermine privacy or create security vulnerabilities?",
     "membership inference"),
]


def targets_for(chunks, phrase):
    p = norm(phrase)
    return [i for i, c in enumerate(chunks) if p in norm(c["text"])]


def main():
    pdf = sys.argv[1]
    chunks = chunk_pdf(pdf)
    print(f"{len(chunks)} chunks")
    if sys.argv[2] == "--check":
        for cat, q, ph in CANDIDATES:
            hits = targets_for(chunks, ph)
            print(f"{len(hits):2d} hits  {cat:12s} {ph[:60]!r} -> {[chunks[i]['chunk_id'] for i in hits][:6]}")
        return

    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import os
    bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=os.environ.get("BI_CACHE"))
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", cache_dir=os.environ.get("CE_CACHE"))
    emb = bi.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64)
    rows = []
    for cat, q, ph in CANDIDATES:
        tgt = set(targets_for(chunks, ph))
        sims = emb @ bi.encode([q], normalize_embeddings=True)[0]
        order = list(np.argsort(-sims))
        top10 = order[:10]
        ce_scores = ce.predict([[q, chunks[i]["text"]] for i in top10])
        reranked = [top10[j] for j in np.argsort(-ce_scores)]
        bi_rank = min((order.index(i) + 1 for i in tgt), default=None)
        ce_rank = min((reranked.index(i) + 1 for i in tgt if i in top10), default=None)
        verdict = ("PROMOTED" if bi_rank and 4 <= bi_rank <= 10 and ce_rank and ce_rank <= 3 else
                   "already_top3" if bi_rank and bi_rank <= 3 and ce_rank and ce_rank <= 3 else
                   "DEMOTED" if bi_rank and bi_rank <= 3 and (not ce_rank or ce_rank > 3) else
                   "stage1_miss" if not bi_rank or bi_rank > 10 else "not_promoted")
        rows.append({"category": cat, "query": q, "target_phrase": ph,
                     "target_chunks": [chunks[i]["chunk_id"] for i in sorted(tgt)],
                     "bi_rank": bi_rank, "ce_rank": ce_rank, "verdict": verdict,
                     "bi_top3": [chunks[i]["chunk_id"] for i in top10[:3]],
                     "ce_top3": [chunks[i]["chunk_id"] for i in reranked[:3]]})
        print(f"{verdict:13s} bi={bi_rank} ce={ce_rank} [{cat}] {q}")
    json.dump(rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
