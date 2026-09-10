# Main experiment results

## Main metrics

| Metric | Result | 95% Wilson CI |
|---|---:|---:|
| Clean RAG accuracy | 77/100 (77.00%) | [67.85%, 84.16%] |
| Poisoned RAG accuracy | 17/100 (17.00%) | [10.89%, 25.55%] |
| Poison Retrieval ASR@1 | 84/100 (84.00%) | [75.58%, 89.90%] |
| Poison Retrieval ASR@5 | 100/100 (100.00%) | [96.30%, 100.00%] |
| Raw generation ASR | 83/100 (83.00%) | [74.45%, 89.11%] |
| Attack-induced ASR | 60/77 (77.92%) | [67.46%, 85.73%] |

## Paired comparison

- Accuracy drop: 60.00%
- Bootstrap 95% CI for accuracy drop: [51.00%, 69.00%]
- Discordant pairs (clean-only / poisoned-only correct): 60 / 0
- Exact McNemar p-value: 1.73472e-18

## Stratified results

| Group | Value | N | Clean accuracy | Poisoned accuracy | Raw ASR | Attack-induced ASR |
|---|---|---:|---:|---:|---:|---:|
| Gold label | yes | 50 | 70.00% | 0.00% | 100.00% | 35/35 (100.00%) |
| Gold label | no | 50 | 84.00% | 34.00% | 66.00% | 25/42 (59.52%) |
| Target poison rank | rank = 1 | 84 | 76.19% | 14.29% | 85.71% | 52/64 (81.25%) |
| Target poison rank | rank > 1 | 16 | 81.25% | 31.25% | 68.75% | 8/13 (61.54%) |

## Provenance defense

| Defense | Detection F1 | Poison ASR@1 | End-to-end accuracy | Utility drop | Attack-induced ASR |
|---|---:|---:|---:|---:|---:|
| no_defense | 0.00% | 84.00% | 17.00% | 60.00% | 77.92% |
| pmid_allowlist | 0.00% | 84.00% | 17.00% | 60.00% | 77.92% |
| content_hash_verification | 100.00% | 0.00% | 77.00% | 0.00% | 0.00% |

## Interpretation

A single passage-only poison document per target question substantially reduced RAG accuracy. The PMID allowlist did not improve security because an attacker could claim an existing PMID. Exact content-hash verification removed all synthetic poisons and restored clean utility under the trusted-canonical-copy assumption.

## Important limitation

Content-hash verification protects document integrity, but it does not detect incorrect, retracted, low-quality, or conflicting claims already present in authentic PubMed records.
