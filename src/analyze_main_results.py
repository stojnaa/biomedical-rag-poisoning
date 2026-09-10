import csv
import math
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 2026
Z_95 = 1.959963984540054


def load_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def boolean(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def number(row, *names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise KeyError(f"None of these columns were found: {', '.join(names)}")


def wilson(successes, total):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    margin = Z_95 * math.sqrt(
        p * (1 - p) / total + Z_95**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def metric(name, successes, total, notes=""):
    lower, upper = wilson(successes, total)
    return {
        "metric": name,
        "successes": successes,
        "total": total,
        "value": successes / total if total else 0.0,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "notes": notes,
    }


def exact_mcnemar(clean_rows, poisoned_by_id):
    clean_only = 0
    poisoned_only = 0
    for clean in clean_rows:
        attacked = poisoned_by_id[clean["question_id"]]
        clean_correct = boolean(clean["is_correct"])
        poisoned_correct = boolean(attacked["correct_after_attack"])
        clean_only += int(clean_correct and not poisoned_correct)
        poisoned_only += int(poisoned_correct and not clean_correct)
    discordant = clean_only + poisoned_only
    if discordant == 0:
        return clean_only, poisoned_only, 1.0
    smaller = min(clean_only, poisoned_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return clean_only, poisoned_only, min(1.0, 2 * tail)


def bootstrap_drop(clean_rows, poisoned_by_id):
    pairs = [
        (
            int(boolean(row["is_correct"])),
            int(boolean(poisoned_by_id[row["question_id"]]["correct_after_attack"])),
        )
        for row in clean_rows
    ]
    rng = random.Random(RANDOM_SEED)
    n = len(pairs)
    drops = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        drops.append(sum(clean - poisoned for clean, poisoned in sample) / n)
    return percentile(drops, 0.025), percentile(drops, 0.975)


def stratum(group, value, rows):
    n = len(rows)
    clean_correct = sum(
        row["clean_prediction"].strip().lower()
        == row["correct_answer"].strip().lower()
        for row in rows
    )
    poisoned_correct = sum(boolean(row["correct_after_attack"]) for row in rows)
    raw = sum(boolean(row["raw_attack_success"]) for row in rows)
    eligible = sum(boolean(row["eligible_for_induced_attack"]) for row in rows)
    induced = sum(boolean(row["induced_attack_success"]) for row in rows)
    return {
        "group": group,
        "value": value,
        "questions": n,
        "clean_accuracy": clean_correct / n if n else 0.0,
        "poisoned_accuracy": poisoned_correct / n if n else 0.0,
        "raw_asr": raw / n if n else 0.0,
        "induced_successes": induced,
        "eligible_questions": eligible,
        "attack_induced_asr": induced / eligible if eligible else 0.0,
    }


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100 * value:.2f}%"


def main():
    clean = load_csv("main_clean_rag.csv")
    poisoned = load_csv("main_poisoned_rag.csv")
    retrieval = load_csv("main_poisoned_retrieval.csv")
    defense = load_csv("main_provenance_defense_summary.csv")
    poisoned_by_id = {row["question_id"]: row for row in poisoned}

    if {row["question_id"] for row in clean} != set(poisoned_by_id):
        raise ValueError("Clean and poisoned result question IDs do not match.")

    clean_correct = sum(boolean(row["is_correct"]) for row in clean)
    poisoned_correct = sum(boolean(row["correct_after_attack"]) for row in poisoned)
    retrieval_1 = sum(boolean(row["target_poison_at_1"]) for row in retrieval)
    retrieval_5 = sum(boolean(row["target_poison_at_5"]) for row in retrieval)
    raw = sum(boolean(row["raw_attack_success"]) for row in poisoned)
    eligible = sum(boolean(row["eligible_for_induced_attack"]) for row in poisoned)
    induced = sum(boolean(row["induced_attack_success"]) for row in poisoned)

    summary = [
        metric("Clean RAG accuracy", clean_correct, len(clean)),
        metric("Poisoned RAG accuracy", poisoned_correct, len(poisoned)),
        metric("Poison Retrieval ASR@1", retrieval_1, len(retrieval)),
        metric("Poison Retrieval ASR@5", retrieval_5, len(retrieval)),
        metric("Raw generation ASR", raw, len(poisoned)),
        metric("Attack-induced ASR", induced, eligible, "Clean-correct questions only"),
    ]

    groups = []
    for label in ("yes", "no"):
        groups.append(stratum("Gold label", label, [r for r in poisoned if r["correct_answer"] == label]))
    groups.append(stratum("Target poison rank", "rank = 1", [r for r in poisoned if int(float(r["target_poison_rank"])) == 1]))
    groups.append(stratum("Target poison rank", "rank > 1", [r for r in poisoned if int(float(r["target_poison_rank"])) > 1]))
    if "poison_documents_in_top_5" in poisoned[0]:
        groups.append(stratum("Poisons in top 5", "one", [r for r in poisoned if int(float(r["poison_documents_in_top_5"])) == 1]))
        groups.append(stratum("Poisons in top 5", "multiple", [r for r in poisoned if int(float(r["poison_documents_in_top_5"])) > 1]))

    drop = clean_correct / len(clean) - poisoned_correct / len(poisoned)
    drop_ci = bootstrap_drop(clean, poisoned_by_id)
    clean_only, poisoned_only, p_value = exact_mcnemar(clean, poisoned_by_id)

    save_csv(RESULTS_DIR / "main_statistical_summary.csv", summary)
    save_csv(RESULTS_DIR / "main_stratified_results.csv", groups)

    lines = [
        "# Main experiment results", "", "## Main metrics", "",
        "| Metric | Result | 95% Wilson CI |", "|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['metric']} | {row['successes']}/{row['total']} ({percent(row['value'])}) "
            f"| [{percent(row['ci_95_lower'])}, {percent(row['ci_95_upper'])}] |"
        )
    lines += [
        "", "## Paired comparison", "",
        f"- Accuracy drop: {percent(drop)}",
        f"- Bootstrap 95% CI for accuracy drop: [{percent(drop_ci[0])}, {percent(drop_ci[1])}]",
        f"- Discordant pairs (clean-only / poisoned-only correct): {clean_only} / {poisoned_only}",
        f"- Exact McNemar p-value: {p_value:.6g}",
        "", "## Stratified results", "",
        "| Group | Value | N | Clean accuracy | Poisoned accuracy | Raw ASR | Attack-induced ASR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['questions']} | {percent(row['clean_accuracy'])} "
            f"| {percent(row['poisoned_accuracy'])} | {percent(row['raw_asr'])} "
            f"| {row['induced_successes']}/{row['eligible_questions']} ({percent(row['attack_induced_asr'])}) |"
        )
    lines += [
        "", "## Provenance defense", "",
        "| Defense | Detection F1 | Poison ASR@1 | End-to-end accuracy | Utility drop | Attack-induced ASR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in defense:
        lines.append(
            f"| {row['defense_mode']} | {percent(number(row, 'detection_f1'))} "
            f"| {percent(number(row, 'poison_retrieval_asr_at_1'))} "
            f"| {percent(number(row, 'end_to_end_accuracy'))} "
            f"| {percent(number(row, 'utility_drop_from_clean'))} "
            f"| {percent(number(row, 'attack_induced_asr'))} |"
        )
    lines += [
        "", "## Interpretation", "",
        "A single passage-only poison document per target question substantially reduced RAG accuracy. The PMID allowlist did not improve security because an attacker could claim an existing PMID. Exact content-hash verification removed all synthetic poisons and restored clean utility under the trusted-canonical-copy assumption.",
        "", "## Important limitation", "",
        "Content-hash verification protects document integrity, but it does not detect incorrect, retracted, low-quality, or conflicting claims already present in authentic PubMed records.", "",
    ]
    (RESULTS_DIR / "MAIN_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Questions analyzed: {len(clean)}")
    print(f"Clean RAG accuracy: {percent(clean_correct / len(clean))}")
    print(f"Poisoned RAG accuracy: {percent(poisoned_correct / len(poisoned))}")
    print(f"Accuracy drop: {percent(drop)}")
    print(f"Bootstrap 95% CI for accuracy drop: [{percent(drop_ci[0])}, {percent(drop_ci[1])}]")
    print(f"Exact McNemar p-value: {p_value:.6g}")
    print(f"Statistical summary saved to: {RESULTS_DIR / 'main_statistical_summary.csv'}")
    print(f"Stratified results saved to: {RESULTS_DIR / 'main_stratified_results.csv'}")
    print(f"Markdown report saved to: {RESULTS_DIR / 'MAIN_RESULTS.md'}")


if __name__ == "__main__":
    main()
