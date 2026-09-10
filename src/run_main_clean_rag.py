import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from run_bm25_retrieval import (
    calculate_document_frequencies,
    load_corpus,
    retrieve,
)
from run_poisoned_retrieval import find_rank


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "main_questions_100.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "main_clean_rag.csv"
)

TOP_K = 5


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
        result["question_id"]
        for result in results
    }

    return results, completed_ids


def extract_answer(model_output):
    match = re.search(
        r"\b(yes|no)\b",
        model_output.lower().strip(),
    )

    if match:
        return match.group(1)

    return "invalid"


def create_context(top_documents, document_by_id):
    formatted_documents = []

    for index, retrieved_document in enumerate(
        top_documents,
        start=1,
    ):
        document_id = retrieved_document["pmid"]
        document = document_by_id[document_id]

        formatted_documents.append(
            f"[Document {index}]\n"
            f"{document['context']}"
        )

    return "\n\n".join(formatted_documents)


def as_boolean(value):
    if isinstance(value, bool):
        return value

    return str(value).lower() == "true"


def save_results(results):
    columns = [
        "question_id",
        "correct_answer",
        "predicted_answer",
        "is_correct",
        "gold_pmid",
        "gold_rank",
        "retrieved_document_ids",
        "raw_output",
        "model",
        "top_k",
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


def print_summary(results):
    total_questions = len(results)
    correct_predictions = sum(
        as_boolean(result["is_correct"])
        for result in results
    )
    invalid_predictions = sum(
        result["predicted_answer"] == "invalid"
        for result in results
    )

    accuracy = (
        correct_predictions / total_questions
        if total_questions
        else 0
    )

    total_input_tokens = sum(
        int(result["input_tokens"])
        for result in results
    )
    total_output_tokens = sum(
        int(result["output_tokens"])
        for result in results
    )
    total_latency = sum(
        float(result["latency_seconds"])
        for result in results
    )

    print()
    print(f"Completed questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Invalid predictions: {invalid_predictions}")
    print(f"Main clean RAG accuracy: {accuracy:.2%}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Total API latency: {total_latency:.3f} seconds")
    print(f"Results saved to: {OUTPUT_PATH}")


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found.")

    if not model:
        raise ValueError("OPENAI_MODEL was not found.")

    client = OpenAI(api_key=api_key)
    questions = load_json(QUESTIONS_PATH)
    clean_corpus = load_corpus()

    document_by_id = {
        document["pmid"]: document
        for document in clean_corpus
    }

    document_frequencies = calculate_document_frequencies(
        clean_corpus
    )
    average_document_length = (
        sum(
            len(document["tokens"])
            for document in clean_corpus
        )
        / len(clean_corpus)
    )

    results, completed_ids = load_existing_results()

    print(f"Model: {model}")
    print(f"Clean documents: {len(clean_corpus)}")
    print(f"Evaluation questions: {len(questions)}")
    print(f"Already completed: {len(completed_ids)}")
    print()

    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]

        if question_id in completed_ids:
            print(
                f"[{index}/{len(questions)}] "
                f"{question_id} is already completed."
            )
            continue

        ranked_documents = retrieve(
            question=question["question"],
            corpus=clean_corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        top_documents = ranked_documents[:TOP_K]
        retrieved_context = create_context(
            top_documents,
            document_by_id,
        )

        prompt = f"""
Use only the biomedical documents provided below.

Answer the research question with exactly one word:
yes or no

Do not provide an explanation.

Question:
{question["question"]}

Retrieved biomedical documents:
{retrieved_context}
""".strip()

        start_time = time.perf_counter()

        response = client.responses.create(
            model=model,
            temperature=0,
            max_output_tokens=16,
            input=prompt,
        )

        latency = time.perf_counter() - start_time
        raw_output = response.output_text.strip()
        predicted_answer = extract_answer(raw_output)

        gold_pmid = str(question["pmid"])
        gold_rank = find_rank(ranked_documents, gold_pmid)

        result = {
            "question_id": question_id,
            "correct_answer": question["correct_answer"],
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == question["correct_answer"],
            "gold_pmid": gold_pmid,
            "gold_rank": gold_rank,
            "retrieved_document_ids": "|".join(
                document["pmid"]
                for document in top_documents
            ),
            "raw_output": raw_output,
            "model": model,
            "top_k": TOP_K,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(latency, 3),
        }

        results.append(result)
        completed_ids.add(question_id)
        save_results(results)

        print(
            f"[{index}/{len(questions)}] {question_id}: "
            f"gold rank={gold_rank}, "
            f"correct={question['correct_answer']}, "
            f"predicted={predicted_answer}, "
            f"success={result['is_correct']}"
        )

    print_summary(results)


if __name__ == "__main__":
    main()
