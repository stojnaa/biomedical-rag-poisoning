"""

HOW TO RUN
1. pip install sentence-transformers --break-system-packages
2. Run from the repo root: python src/dense_retriever_check.py

"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

QUESTIONS_PATH = "data/processed/main_questions_100.json"
POISON_PATH = "data/poisoned/main_poison_documents_100.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    model = SentenceTransformer(MODEL_NAME)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(POISON_PATH, encoding="utf-8") as f:
        poisons = json.load(f)

    poison_by_pmid = {p["target_pmid"]: p for p in poisons}

    # Build the 200-document corpus: one clean doc + one poison doc per question.
    doc_ids = []     # ("clean", pmid) or ("poison", pmid)
    doc_texts = []
    for q in questions:
        pmid = q["pmid"]
        doc_ids.append(("clean", pmid))
        doc_texts.append(q["context"])
        if pmid in poison_by_pmid:
            doc_ids.append(("poison", pmid))
            doc_texts.append(poison_by_pmid[pmid]["generated_passage"])

    print(f"Embedding {len(doc_texts)} documents "
          f"({sum(1 for k,_ in doc_ids if k=='clean')} clean + "
          f"{sum(1 for k,_ in doc_ids if k=='poison')} poison)...")
    doc_embeddings = model.encode(doc_texts, batch_size=64, show_progress_bar=True,
                                   normalize_embeddings=True)

    poison_row_by_pmid = {
        pmid: i for i, (kind, pmid) in enumerate(doc_ids) if kind == "poison"
    }

    hits_at_k = {k: 0 for k in range(1, TOP_K + 1)}
    n_eval = 0

    for q in questions:
        pmid = q["pmid"]
        if pmid not in poison_row_by_pmid:
            continue
        n_eval += 1

        q_embedding = model.encode([q["question"]], normalize_embeddings=True)[0]
        scores = doc_embeddings @ q_embedding
        ranked_idx = np.argsort(-scores)[:TOP_K]

        target_row = poison_row_by_pmid[pmid]
        rank = None
        for pos, idx in enumerate(ranked_idx, start=1):
            if idx == target_row:
                rank = pos
                break

        if rank is not None:
            for k in range(rank, TOP_K + 1):
                hits_at_k[k] += 1

    print(f"\nEvaluated {n_eval} questions over a {len(doc_texts)}-document "
          f"corpus (100 clean target docs + 100 poison passages) with "
          f"{MODEL_NAME}.\n")
    print("=== Poison Retrieval ASR@k (dense retriever, 200-doc corpus) ===")
    for k in range(1, TOP_K + 1):
        asr = 100 * hits_at_k[k] / n_eval if n_eval else 0
        print(f"ASR@{k}: {asr:.1f}%  ({hits_at_k[k]}/{n_eval})")


if __name__ == "__main__":
    main()
