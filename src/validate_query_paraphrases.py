import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARAPHRASES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_query_paraphrases.json"
)

REVIEWED_CORRECTIONS = {
    "pilot_04": (
        "Does adding an antirotation U-Blade (RC) lag screw "
        "improve treatment outcomes for AO/OTA 31 A1-3 fractures "
        "treated with a gamma 3 nail?"
    ),
    "pilot_14": (
        "Are sugar-free medicines more erosive than "
        "sugar-containing medicines?"
    ),
}


def main():
    with PARAPHRASES_PATH.open("r", encoding="utf-8") as file:
        paraphrases = json.load(file)

    question_ids = {
        item["question_id"]
        for item in paraphrases
    }

    missing_ids = set(REVIEWED_CORRECTIONS) - question_ids

    if missing_ids:
        raise ValueError(
            "The following reviewed questions are missing: "
            + ", ".join(sorted(missing_ids))
        )

    for item in paraphrases:
        question_id = item["question_id"]

        if question_id in REVIEWED_CORRECTIONS:
            item["paraphrased_question"] = (
                REVIEWED_CORRECTIONS[question_id]
            )
            item["review_action"] = "corrected"
        else:
            item["review_action"] = "approved_without_changes"

        item["manually_validated"] = True

    with PARAPHRASES_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            paraphrases,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Validated paraphrases: {len(paraphrases)}")
    print(f"Corrected paraphrases: {len(REVIEWED_CORRECTIONS)}")
    print(f"Updated file: {PARAPHRASES_PATH}")


if __name__ == "__main__":
    main()
