# Reproducibility guide

## Environment

- Recommended Python: 3.11 or 3.12
- Retriever: BM25
- Retrieval depth: 5
- Generator: `gpt-4o-mini-2024-07-18`
- Generation temperature: 0
- Main evaluation seed: 2026
- Main evaluation size: 100 questions, balanced 50 `yes` and 50 `no`

## Full execution order

The first command downloads the original PubMedQA labeled dataset. Commands marked **API** require an OpenAI API key and incur usage charges.

```powershell
python src\download_pubmedqa.py
python src\prepare_pilot.py
python src\run_oracle_baseline.py                 # API
python src\run_bm25_retrieval.py
python src\run_clean_rag.py                       # API
python src\generate_poison_documents.py           # API
python src\run_poisoned_retrieval.py
python src\run_poisoned_rag.py                    # API
python src\run_attack_ablation.py
python src\run_passage_only_poisoned_rag.py       # API
python src\generate_query_paraphrases.py           # API
python src\validate_query_paraphrases.py
python src\run_paraphrased_query_retrieval.py
python src\run_paraphrased_query_clean_rag.py      # API
python src\run_paraphrased_query_poisoned_rag.py   # API
python src\run_provenance_ablation.py
python src\prepare_main_evaluation.py
python src\run_main_bm25_retrieval.py
python src\run_main_clean_rag.py                   # API
python src\generate_main_poison_documents.py       # API
python src\run_main_poisoned_retrieval.py
python src\run_main_poisoned_rag.py                # API
python src\run_main_provenance_defense.py
python src\analyze_main_results.py
python src\create_publication_artifacts.py
python src\validate_project.py
```

All API scripts save progress incrementally. Existing output files should be preserved if execution is interrupted.

## Metric definitions

- **Recall@k:** proportion of questions for which the original relevant PubMedQA document appears in the top `k` results.
- **Retrieval ASR@k:** proportion of questions for which the targeted poison document appears in the top `k` results.
- **Raw ASR:** proportion of all questions for which the poisoned RAG output equals the attacker's target answer.
- **Attack-induced ASR:** proportion of eligible questions for which the attack changes a non-target clean prediction to the target answer.
- **Utility drop:** clean RAG accuracy minus accuracy under the evaluated attack or defense configuration.
- **Detection precision, recall, and F1:** poison-document detection metrics.

## Determinism and retained artifacts

The evaluation split is fixed with seed 2026 and committed in `data/processed/main_questions_100.json`. Generated poison passages and per-question model outputs are committed so the reported analysis can be reproduced without repeating paid or potentially nondeterministic API calls.

## Validation

`src/validate_project.py` checks row counts, ID alignment, label balance, headline metrics, and the principal defense result. A passing validation confirms internal consistency of the committed artifacts; it does not independently replicate the model API calls.
