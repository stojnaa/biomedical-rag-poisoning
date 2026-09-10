import csv
import json
import random
from collections import Counter
from pathlib import Path


RANDOM_SEED = 2026
QUESTIONS_PER_LABEL = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ori_pqal.json"
)

PILOT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "main_questions_100.csv"
)

JSON_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "main_questions_100.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def convert_to_records(dataset):
    records = []

    for pmid, item in dataset.items():
        contexts = item.get("CONTEXTS", [])

        records.append(
            {
                "pmid": str(pmid),
                "question": item.get("QUESTION", "").strip(),
                "context": " ".join(
                    context.strip()
                    for context in contexts
                ),
                "long_answer": item.get("LONG_ANSWER", "").strip(),
                "correct_answer": (
                    item.get("final_decision", "").lower().strip()
                ),
                "year": item.get("YEAR", ""),
            }
        )

    return records


def create_balanced_evaluation_set(records, excluded_pmids):
    random_generator = random.Random(RANDOM_SEED)

    records_by_label = {
        "yes": [],
        "no": [],
    }

    for record in records:
        label = record["correct_answer"]

        if (
            label in records_by_label
            and record["pmid"] not in excluded_pmids
            and record["question"]
            and record["context"]
        ):
            records_by_label[label].append(record)

    selected_records = []

    for label in ("yes", "no"):
        candidates = sorted(
            records_by_label[label],
            key=lambda record: record["pmid"],
        )

        if len(candidates) < QUESTIONS_PER_LABEL:
            raise ValueError(
                f"Not enough {label} candidates. "
                f"Required: {QUESTIONS_PER_LABEL}, "
                f"available: {len(candidates)}"
            )

        selected_records.extend(
            random_generator.sample(
                candidates,
                QUESTIONS_PER_LABEL,
            )
        )

    random_generator.shuffle(selected_records)

    for index, record in enumerate(selected_records, start=1):
        record["question_id"] = f"main_{index:03d}"

    return selected_records


def save_as_csv(records):
    columns = [
        "question_id",
        "pmid",
        "year",
        "question",
        "context",
        "long_answer",
        "correct_answer",
    ]

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def save_as_json(records):
    JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    raw_dataset = load_json(RAW_DATASET_PATH)
    pilot_questions = load_json(PILOT_PATH)
    records = convert_to_records(raw_dataset)

    pilot_pmids = {
        str(item["pmid"])
        for item in pilot_questions
    }

    evaluation_records = create_balanced_evaluation_set(
        records,
        pilot_pmids,
    )

    save_as_csv(evaluation_records)
    save_as_json(evaluation_records)

    selected_pmids = {
        record["pmid"]
        for record in evaluation_records
    }

    overlap = selected_pmids & pilot_pmids
    label_distribution = Counter(
        record["correct_answer"]
        for record in evaluation_records
    )

    print(f"Total PubMedQA questions: {len(records)}")
    print(f"Pilot questions excluded: {len(pilot_pmids)}")
    print(f"Main evaluation questions: {len(evaluation_records)}")
    print(f"Label distribution: {dict(label_distribution)}")
    print(f"PMID overlap with pilot set: {len(overlap)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"CSV saved to: {CSV_OUTPUT_PATH}")
    print(f"JSON saved to: {JSON_OUTPUT_PATH}")
    print()
    print("First three evaluation questions:")

    for record in evaluation_records[:3]:
        print("-" * 60)
        print(f"ID: {record['question_id']}")
        print(f"PMID: {record['pmid']}")
        print(f"Question: {record['question']}")
        print(f"Correct answer: {record['correct_answer']}")


if __name__ == "__main__":
    main()
