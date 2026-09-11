import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#059669"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"


def load_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required results file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_figure(figure, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LIGHT_GRAY)
    axis.spines["bottom"].set_color(LIGHT_GRAY)
    axis.tick_params(colors="#334155")
    axis.set_ylim(0, 112)
    axis.set_ylabel("Percentage (%)")
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    axis.set_axisbelow(True)


def add_labels(axis, bars):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def create_accuracy_figure(summary):
    values = {
        row["metric"]: 100 * float(row["value"])
        for row in summary
    }
    labels = ["Clean RAG", "Poisoned RAG"]
    heights = [values["Clean RAG accuracy"], values["Poisoned RAG accuracy"]]

    figure, axis = plt.subplots(figsize=(5.4, 3.7))
    bars = axis.bar(labels, heights, color=[BLUE, RED], width=0.58)
    style_axis(axis)
    axis.set_title("End-to-end RAG accuracy", fontweight="bold", pad=12)
    add_labels(axis, bars)
    figure.tight_layout()
    save_figure(figure, "fig1_clean_vs_poisoned_accuracy")


def create_attack_figure(summary):
    values = {
        row["metric"]: 100 * float(row["value"])
        for row in summary
    }
    labels = ["Retrieval\nASR@1", "Retrieval\nASR@5", "Raw\nASR", "Attack-induced\nASR"]
    heights = [
        values["Poison Retrieval ASR@1"],
        values["Poison Retrieval ASR@5"],
        values["Raw generation ASR"],
        values["Attack-induced ASR"],
    ]

    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    bars = axis.bar(labels, heights, color=[GRAY, GRAY, RED, RED], width=0.62)
    style_axis(axis)
    axis.set_title("Attack effectiveness", fontweight="bold", pad=12)
    add_labels(axis, bars)
    figure.tight_layout()
    save_figure(figure, "fig2_attack_effectiveness")


def create_defense_figure(defense):
    display_names = {
        "no_defense": "No defense",
        "pmid_allowlist": "PMID allowlist",
        "content_hash_verification": "Content hash",
    }
    labels = [display_names[row["defense_mode"]] for row in defense]
    accuracies = [100 * float(row["end_to_end_accuracy"]) for row in defense]
    attack_rates = [100 * float(row["attack_induced_asr"]) for row in defense]
    x_positions = list(range(len(labels)))
    width = 0.35

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    accuracy_bars = axis.bar(
        [x - width / 2 for x in x_positions],
        accuracies,
        width,
        label="End-to-end accuracy",
        color=GREEN,
    )
    attack_bars = axis.bar(
        [x + width / 2 for x in x_positions],
        attack_rates,
        width,
        label="Attack-induced ASR",
        color=RED,
    )
    style_axis(axis)
    axis.set_title("Provenance defense comparison", fontweight="bold", pad=12)
    axis.set_xticks(x_positions, labels)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    add_labels(axis, accuracy_bars)
    add_labels(axis, attack_bars)
    figure.tight_layout()
    save_figure(figure, "fig3_provenance_defense")


def escape_latex(text):
    return (
        str(text)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("@", "@")
    )


def create_latex_tables(summary, stratified, defense):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    main_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main attack results on 100 held-out PubMedQA questions.}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Metric & Result & 95\\% CI \\\\",
        "\\hline",
    ]
    for row in summary:
        result = f"{100 * float(row['value']):.1f}\\%"
        interval = (
            f"[{100 * float(row['ci_95_lower']):.1f}, "
            f"{100 * float(row['ci_95_upper']):.1f}]\\%"
        )
        main_lines.append(f"{escape_latex(row['metric'])} & {result} & {interval} \\\\")
    main_lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    (TABLES_DIR / "table_main_results.tex").write_text(
        "\n".join(main_lines), encoding="utf-8"
    )

    stratified_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Attack results stratified by label and retrieval conditions.}",
        "\\label{tab:stratified-results}",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Group & Value & $N$ & Raw ASR & Induced ASR \\\\",
        "\\hline",
    ]
    for row in stratified:
        stratified_lines.append(
            f"{escape_latex(row['group'])} & {escape_latex(row['value'])} & "
            f"{row['questions']} & {100 * float(row['raw_asr']):.1f}\\% & "
            f"{100 * float(row['attack_induced_asr']):.1f}\\% \\\\"
        )
    stratified_lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    (TABLES_DIR / "table_stratified_results.tex").write_text(
        "\n".join(stratified_lines), encoding="utf-8"
    )

    defense_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison of provenance defenses.}",
        "\\label{tab:defense-results}",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Defense & Detection F1 & Accuracy & Induced ASR \\\\",
        "\\hline",
    ]
    for row in defense:
        defense_lines.append(
            f"{escape_latex(display_defense(row['defense_mode']))} & "
            f"{100 * float(row['detection_f1']):.1f}\\% & "
            f"{100 * float(row['end_to_end_accuracy']):.1f}\\% & "
            f"{100 * float(row['attack_induced_asr']):.1f}\\% \\\\"
        )
    defense_lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    (TABLES_DIR / "table_defense_results.tex").write_text(
        "\n".join(defense_lines), encoding="utf-8"
    )


def display_defense(mode):
    return {
        "no_defense": "No defense",
        "pmid_allowlist": "PMID allowlist",
        "content_hash_verification": "Content hash",
    }[mode]


def main():
    summary = load_csv("main_statistical_summary.csv")
    stratified = load_csv("main_stratified_results.csv")
    defense = load_csv("main_provenance_defense_summary.csv")

    create_accuracy_figure(summary)
    create_attack_figure(summary)
    create_defense_figure(defense)
    create_latex_tables(summary, stratified, defense)

    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"LaTeX tables saved to: {TABLES_DIR}")
    print("Generated 3 figures in PNG and PDF formats.")
    print("Generated 3 LaTeX tables.")


if __name__ == "__main__":
    main()
