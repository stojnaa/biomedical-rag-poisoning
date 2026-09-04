# Pilot experiment results

## Experimental setup

- Dataset: PubMedQA PQA-L
- Evaluation subset: 20 questions
- Labels: 10 yes and 10 no
- Corpus size: 1,000 PubMedQA contexts
- Retriever: BM25
- Top-k: 5
- Generator: gpt-4o-mini-2024-07-18
- Temperature: 0
- Poisoning budget: one targeted poison document per question
- Number of poison documents: 20
- Random seed: 42

## Clean baseline

- Oracle accuracy: 70%
- BM25 Recall@1: 100%
- BM25 Recall@5: 100%
- MRR: 1.0000
- Clean RAG accuracy: 65%

## Poisoning attack

- Poisoned corpus size: 1,020 documents
- Poison retrieval ASR@1: 100%
- Poison retrieval ASR@5: 100%
- Original document Recall@1: 0%
- Original document Recall@5: 100%
- Poisoned RAG accuracy: 5%
- Raw ASR: 95%
- Attack-induced ASR: 92.31%
- Changed predictions: 12/20

## Provenance defense

- Detected poison documents: 20/20
- False positives: 0/1,000
- False negatives: 0/20
- Precision: 100%
- Recall: 100%
- F1: 100%
- False-positive rate: 0%
- Defense latency: 0.091521 seconds

## Preliminary conclusion

A single targeted poison document per question successfully
hijacked BM25 retrieval and changed 12 of the 13 answers that
were correct in the clean RAG configuration.

The provenance defense removed all poison documents without
removing legitimate documents. However, this defense currently
detects only documents with missing or falsified provenance and
modified content. It does not assess the credibility or scientific
validity of authentic PubMed sources.

## Pilot limitations

- Only 20 evaluation questions
- Only yes/no questions
- Only one retriever
- Only one LLM
- Small corpus of 1,000 documents
- Target questions are known to the attacker
- Questions substantially overlap with corresponding abstracts
- Poison documents contain the exact target question
- Synthetic poison ratio is approximately 1.96%
- Exact-content provenance verification assumes access to a
  trusted canonical copy