# Biomedical RAG Knowledge-Poisoning Evaluation

This repository contains a controlled academic evaluation of targeted knowledge-poisoning attacks against a biomedical retrieval-augmented generation (RAG) pipeline. It uses the labeled PubMedQA subset, BM25 retrieval, and `gpt-4o-mini-2024-07-18` for binary question answering.

The project measures retrieval compromise, end-to-end answer manipulation, and the effectiveness of provenance checks. Generated poison passages are included only as research artifacts for reproducibility.

## Main findings

- Clean BM25 Recall@1: **98%**
- Clean RAG accuracy: **77%**
- Poison retrieval ASR@1: **84%**
- Poison retrieval ASR@5: **100%**
- Poisoned RAG accuracy: **17%**
- Attack-induced ASR: **60/77 (77.92%)**
- Accuracy drop: **60 percentage points**, bootstrap 95% CI **[51, 69]**
- Exact McNemar test: **p < 0.001**
- Content-hash verification: **100% detection F1** and **0% measured attack-induced ASR** in this controlled setting

The PMID allowlist did not detect the poison passages because they claimed identifiers that already existed. Exact content-hash verification succeeded because the experiment assumed access to a trusted canonical PubMedQA corpus.

## Repository structure

```text
data/processed/     Fixed pilot and held-out evaluation questions
data/poisoned/      Synthetic counterfactual poison passages
figures/            Publication figures in PNG and PDF formats
results/            Per-question outputs and statistical summaries
src/                Data preparation, experiments, analysis, and validation
tables/             LaTeX tables generated from result CSV files
```

Raw PubMedQA data are excluded from Git and can be downloaded with the supplied script.

## Setup

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your own API key in `.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini-2024-07-18
```

Never commit `.env`. API calls incur usage charges. Analysis, validation, retrieval, and figure generation do not call the OpenAI API.

## Reproduce the main experiment

Run the commands from the repository root. The complete sequence is documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

To verify the committed results without making API calls:

```powershell
python src\analyze_main_results.py
python src\create_publication_artifacts.py
python src\validate_project.py
```

## Experimental design

The main evaluation contains 100 held-out PubMedQA questions, balanced between `yes` and `no`, with no PMID overlap with the 20-question pilot. The attacker adds one synthetic passage-only poison document per target question. Each passage supports the opposite answer without repeating the exact question. Retrieval uses BM25 with `top_k=5`; the generator must return exactly `yes` or `no`.

Attack-induced ASR is calculated only for questions whose clean prediction did not already equal the attacker's target answer. Wilson intervals are reported for proportions, a paired bootstrap interval is reported for the accuracy drop, and an exact McNemar test compares clean and poisoned correctness.

## Scope and limitations

- One sparse retriever, one LLM, one corpus, and binary questions were evaluated.
- The attacker knew the target topics and added one poison document per question.
- The generated passages were synthetic and were not inserted into PubMed or any production system.
- Exact content-hash verification assumes a trusted canonical copy of every approved document.
- Hash verification protects integrity, not scientific truth. It cannot identify authentic but incorrect, retracted, low-quality, or conflicting evidence.
- Perfect defense results refer only to this controlled experimental setting and should not be generalized to open biomedical retrieval systems.

## Research ethics

This repository supports defensive, controlled research. Do not inject generated passages into real biomedical databases, search indexes, clinical tools, or systems used for patient care.

## Authors

Stojna Pušičić and Marta Runtić, University of Belgrade, School of Electrical Engineering.
