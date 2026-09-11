import csv
import json
import random
from collections import Counter
from pathlib import Path


RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ori_pqal.json"

CSV_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.csv"
)

JSON_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)


def load_pubmedqa():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def convert_to_records(dataset):
    records = []

    for pmid, item in dataset.items():
        contexts = item.get("CONTEXTS", [])

        record = {
            "pmid": str(pmid),
            "question": item.get("QUESTION", "").strip(),
            "context": " ".join(
                context.strip() for context in contexts
            ),
            "long_answer": item.get("LONG_ANSWER", "").strip(),
            "correct_answer": (
                item.get("final_decision", "").lower().strip()
            ),
            "year": item.get("YEAR", ""),
        }

        records.append(record)

    return records


def create_balanced_pilot(records):
    random_generator = random.Random(RANDOM_SEED)

    records_by_label = {
        "yes": [],
        "no": [],
    }

    for record in records:
        label = record["correct_answer"]

        if label in records_by_label:
            records_by_label[label].append(record)

    selected_records = []

    for label in ["yes", "no"]:
        group = sorted(
            records_by_label[label],
            key=lambda record: record["pmid"],
        )

        selected_records.extend(
            random_generator.sample(group, 10)
        )

    random_generator.shuffle(selected_records)

    for index, record in enumerate(selected_records, start=1):
        record["question_id"] = f"pilot_{index:02d}"

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

    with CSV_OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def save_as_json(records):
    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    dataset = load_pubmedqa()
    records = convert_to_records(dataset)
    pilot_records = create_balanced_pilot(records)

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_as_csv(pilot_records)
    save_as_json(pilot_records)

    all_labels = Counter(
        record["correct_answer"] for record in records
    )

    pilot_labels = Counter(
        record["correct_answer"] for record in pilot_records
    )

    print("Total PubMedQA questions:", len(records))
    print("Full dataset label distribution:", dict(all_labels))
    print()
    print("Pilot questions:", len(pilot_records))
    print("Pilot label distribution:", dict(pilot_labels))
    print()
    print("CSV:", CSV_OUTPUT_PATH)
    print("JSON:", JSON_OUTPUT_PATH)
    print()
    print("First three questions:")

    for record in pilot_records[:3]:
        print("-" * 60)
        print("ID:", record["question_id"])
        print("PMID:", record["pmid"])
        print("Question:", record["question"])
        print("Correct answer:", record["correct_answer"])


if __name__ == "__main__":
    main()
