# Results interpretation

## Primary result

The controlled passage-only attack reduced accuracy from 77% to 17%. The paired accuracy drop was 60 percentage points, with a bootstrap 95% confidence interval of 51–69 points. All 60 discordant question pairs favored the clean condition, and the exact McNemar test gave `p < 0.001`.

This supports the conclusion that the observed degradation was associated with the poisoning intervention rather than ordinary variation across different questions.

## Retrieval and generation

The target poison appeared at rank one for 84% of questions and within the top five for 100%. Raw generation ASR was 83%, while attack-induced ASR was 77.92% among the 77 eligible questions.

These metrics answer different questions:

- Retrieval ASR measures whether the attack reached the generator context.
- Raw ASR measures whether the final answer matched the attack target.
- Attack-induced ASR excludes questions whose clean answer already matched that target.

The difference between 100% Retrieval ASR@5 and 83% Raw ASR shows that retrieving a poison passage does not guarantee control of the final answer. The clean and poison documents can coexist in the context, and the generator may not follow the targeted claim.

## Subgroup observations

- Gold `yes`: raw ASR 100%, poisoned accuracy 0%, induced ASR 100%.
- Gold `no`: raw ASR 66%, poisoned accuracy 34%, induced ASR 59.52%.
- Poison rank 1: raw ASR 85.71%, induced ASR 81.25%.
- Poison rank greater than 1: raw ASR 68.75%, induced ASR 61.54%.
- One poison in top 5: raw ASR 82.81%, induced ASR 75.56%.
- Multiple poisons in top 5: raw ASR 83.33%, induced ASR 81.25%.

The label asymmetry is substantial, but the subgroups are smaller than the full evaluation. It should be described as an observed pattern rather than a universal model property. A follow-up experiment with repeated sampling or a larger dataset would be needed to establish the mechanism.

## Defense result

The PMID allowlist had the same outcome as no defense because poison passages reused identifiers present in the allowed corpus. An identifier therefore established neither document authenticity nor content integrity.

Exact content-hash verification rejected all 100 synthetic poisons, accepted all 1,000 clean documents, restored 77% end-to-end accuracy, and produced zero measured attack-induced successes. The appropriate claim is:

> Exact content-hash verification achieved perfect detection in this controlled experiment under a trusted-canonical-copy assumption.

The result does not show that content hashing detects biomedical misinformation. It detects unauthorized content changes. Authentic but incorrect, retracted, outdated, low-quality, or conflicting publications remain outside the scope of this defense.

## Claims to avoid

Do not claim that:

- the defense is universally perfect;
- the attack represents every possible RAG poisoning threat;
- the experiment evaluates clinical safety or patient outcomes;
- the balanced 100-question subset represents the natural PubMedQA label distribution;
- statistical significance alone establishes practical generalization to other retrievers or models.
