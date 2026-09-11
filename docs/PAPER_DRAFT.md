# Targeted Knowledge Poisoning in Biomedical Retrieval-Augmented Generation

**Stojna Pušičić, Marta Runtić**  
University of Belgrade, School of Electrical Engineering  

## Abstract

Retrieval-augmented generation can improve biomedical question answering by grounding a language model in retrieved literature, but it also introduces a path for knowledge-poisoning attacks. We evaluate a targeted passage-injection attack against a biomedical RAG pipeline based on BM25, PubMedQA, and GPT-4o mini. The main experiment uses 100 held-out binary questions and adds one synthetic counterfactual passage per question. The poison passage entered the top retrieval position for 84% of questions and the top five for all questions. End-to-end accuracy decreased from 77% to 17%, a 60-percentage-point drop with a paired bootstrap 95% confidence interval of 51–69 points. The attack-induced success rate was 77.92% on questions eligible for a prediction change. A PMID allowlist provided no protection because forged passages reused valid identifiers, whereas exact content-hash verification rejected all synthetic poisons and restored clean accuracy in this controlled setting. These results demonstrate that source identifiers alone are insufficient for biomedical RAG integrity and that cryptographic verification is effective when a trusted canonical corpus is available.

**Keywords:** retrieval-augmented generation, data poisoning, biomedical question answering, provenance, PubMedQA

## 1. Introduction

Biomedical RAG systems combine information retrieval with large language models to ground generated answers in domain documents. This architecture can improve access to specialized evidence, but the retrieved corpus becomes part of the model's effective input and therefore part of its attack surface. A malicious or compromised ingestion pipeline may introduce documents that are lexically relevant to a question while supporting a false conclusion.

Previous research has demonstrated poisoning vulnerabilities in general RAG systems and in medical language models [1]–[3]. Biomedical deployments make this problem especially important because incorrect retrieved evidence can affect health-related answers. We study a narrow but measurable threat: an attacker who can add one synthetic counterfactual passage for each target question but cannot modify the retriever or generator.

Our contributions are: (1) a reproducible targeted poisoning evaluation on a held-out PubMedQA subset; (2) separate measurement of retrieval compromise and end-to-end answer manipulation; (3) evaluation of attack transfer beyond exact-question copying; and (4) an ablation of identifier allowlisting and exact content-hash verification.

## 2. Methodology

### 2.1 Dataset and split

We use the 1,000 labeled examples in PubMedQA PQA-L. A 20-question balanced pilot supported pipeline development. The main evaluation contains 100 different questions, balanced between 50 `yes` and 50 `no` labels. Pilot PMIDs were excluded, and the main subset was sampled with seed 2026. Each PubMedQA context forms one corpus document, producing a clean corpus of 1,000 documents.

### 2.2 RAG pipeline

Documents are ranked using BM25, and the top five contexts are supplied to `gpt-4o-mini-2024-07-18`. The prompt restricts the model to the retrieved biomedical documents and requires exactly one binary answer. Temperature is set to zero. Clean retrieval is evaluated with Recall@1, Recall@5, and mean reciprocal rank; generation is evaluated with exact-match accuracy.

### 2.3 Poisoning attack

For every main-evaluation question, the attacker adds one 80–120-word synthetic passage supporting the opposite label. The prompt requests a neutral biomedical abstract style, prohibits repetition of the exact question, and excludes fabricated bibliographic metadata. The attacked corpus therefore contains 1,100 documents. The attack budget is one targeted document per question.

Retrieval attack success rate at rank $k$ measures whether the target poison enters the top $k$. Raw generation ASR measures how often the attacked answer equals the target label. Attack-induced ASR considers only questions whose clean prediction did not already equal that label, which avoids counting pre-existing errors as successful changes.

### 2.4 Provenance defenses

We compare no defense, a PMID allowlist, and exact SHA-256 content verification against a trusted canonical copy. Allowlisting accepts any document claiming a known PMID. Hash verification accepts a known PMID only if its content hash matches the canonical document.

### 2.5 Statistical analysis

We report Wilson 95% confidence intervals for proportions. The clean-versus-poisoned accuracy drop uses 10,000 paired bootstrap samples with seed 2026. An exact McNemar test evaluates paired correctness changes.

## 3. Results

Clean BM25 retrieval achieved 98% Recall@1, 100% Recall@5, and 0.99 MRR. Clean end-to-end accuracy was 77% (95% CI: 67.85–84.16%). Under attack, the target poison reached rank one for 84% of questions and the top five for all questions. Accuracy fell to 17% (95% CI: 10.89–25.55%), yielding a 60-point drop (paired bootstrap 95% CI: 51–69 points). The exact McNemar test confirmed a significant change ($p<0.001$). Raw ASR was 83%, while attack-induced ASR was 60/77, or 77.92%.

The attack was asymmetric across labels. For gold-`yes` questions, poisoned accuracy was 0% and raw ASR was 100%. For gold-`no` questions, poisoned accuracy was 34% and raw ASR was 66%. When the target poison ranked first, attack-induced ASR was 81.25%; when it ranked below first, it was 61.54%. These subgroup results are descriptive because of their smaller sample sizes.

The PMID allowlist reproduced the undefended result: 17% accuracy and 77.92% attack-induced ASR. Exact content-hash verification rejected all 100 generated poison passages with no clean-document rejection, restored accuracy to 77%, and reduced measured attack-induced ASR to zero.

## 4. Discussion

The results show that a single lexically relevant counterfactual passage can dominate sparse retrieval and strongly influence a grounded language model. The 100% Retrieval ASR@5 indicates that considering several documents does not by itself remove the attack; the generator still receives the poisoned evidence. The difference between rank-one and lower-ranked cases suggests that retrieval position affects influence but does not fully determine it.

Identifier allowlisting failed because an identifier is a claim rather than proof of document integrity. Exact hash verification addressed this failure under a closed-corpus assumption. However, its perfect measured performance should not be interpreted as a universal biomedical misinformation defense. It requires a trusted canonical copy and protects bytes, not scientific validity. Authentic publications may still be incorrect, retracted, low quality, or in conflict with newer evidence.

The study is limited to one sparse retriever, one language model, one binary benchmark, and a targeted attacker who knows the evaluated topics. The balanced main subset is useful for comparison but does not preserve the natural PubMedQA label distribution. Future work should test dense and hybrid retrievers, multiple generators, adaptive paraphrases, multiple-document attack budgets, citation-aware generation, source consensus, retraction metadata, and temporally updated evidence.

## 5. Conclusion

Targeted passage injection caused a statistically significant 60-percentage-point loss in biomedical RAG accuracy. Valid-looking PMIDs alone did not provide provenance security. Exact content verification eliminated the synthetic attack in the controlled corpus, illustrating the value of integrity checks while also exposing the need for defenses that assess the credibility and agreement of authentic sources.

## References

[1] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models,” *34th USENIX Security Symposium*, 2025.

[2] D. A. Alber et al., “Medical large language models are vulnerable to data-poisoning attacks,” *Nature Medicine*, 2025, doi: 10.1038/s41591-024-03445-1.

[3] P. Yang et al., “Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation,” arXiv:2605.10253, 2026.

[4] M. Li, H. Kilicoglu, H. Xu, and R. Zhang, “BiomedRAG: A Retrieval Augmented Large Language Model for Biomedicine,” *Journal of Biomedical Informatics*, vol. 162, 104769, 2025.

[5] R. Yang et al., “Retrieval-augmented generation in medicine: A scoping review of technical implementations, clinical applications, and ethical considerations,” 2026.

> Formatting note: transfer this content into the official TELFOR template and confirm the final author list, affiliations, reference metadata, and page limit with the mentors before submission.
