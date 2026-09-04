import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "oracle_baseline.csv"
)


def extract_answer(model_output):
    cleaned_output = model_output.lower().strip()

    match = re.search(r"\b(yes|no)\b", cleaned_output)

    if match:
        return match.group(1)

    return "invalid"


def load_existing_results():
    if not OUTPUT_PATH.exists():
        return [], set()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        results = list(csv.DictReader(file))

    completed_ids = {
        result["question_id"] for result in results
    }

    return results, completed_ids


def save_results(results):
    columns = [
        "question_id",
        "pmid",
        "correct_answer",
        "predicted_answer",
        "is_correct",
        "raw_output",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY nije pronađen.")

    if not model:
        raise ValueError("OPENAI_MODEL nije pronađen.")

    client = OpenAI(api_key=api_key)

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    results, completed_ids = load_existing_results()

    print("Model:", model)
    print("Ukupno pitanja:", len(questions))
    print("Već završeno:", len(completed_ids))
    print()

    for index, item in enumerate(questions, start=1):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(f"[{index}/20] {question_id} je već obrađen.")
            continue

        prompt = f"""
Use only the biomedical context provided below.

Answer the research question with exactly one word:
yes or no

Do not provide an explanation.

Question:
{item["question"]}

Biomedical context:
{item["context"]}
""".strip()

        start_time = time.perf_counter()

        response = client.responses.create(
            model=model,
            temperature=0,
            input=prompt,
        )

        latency = time.perf_counter() - start_time

        raw_output = response.output_text.strip()
        predicted_answer = extract_answer(raw_output)
        correct_answer = item["correct_answer"]

        is_correct = predicted_answer == correct_answer

        result = {
            "question_id": question_id,
            "pmid": item["pmid"],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "raw_output": raw_output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(latency, 3),
        }

        results.append(result)
        completed_ids.add(question_id)

        save_results(results)

        print(
            f"[{index}/20] {question_id}: "
            f"tačno={correct_answer}, "
            f"model={predicted_answer}, "
            f"uspeh={is_correct}"
        )

    correct_count = sum(
        str(result["is_correct"]).lower() == "true"
        for result in results
    )

    accuracy = correct_count / len(results) if results else 0

    print()
    print("Završena pitanja:", len(results))
    print("Tačni odgovori:", correct_count)
    print(f"Accuracy: {accuracy:.2%}")
    print("Rezultati:", OUTPUT_PATH)


if __name__ == "__main__":
    main()