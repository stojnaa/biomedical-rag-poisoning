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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "clean_rag.csv"
)

TOP_K = 5


def extract_answer(model_output):
    match = re.search(
        r"\b(yes|no)\b",
        model_output.lower().strip(),
    )

    if match:
        return match.group(1)

    return "invalid"


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
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
        result["question_id"] for result in results
    }

    return results, completed_ids


def save_results(results):
    columns = [
        "question_id",
        "relevant_pmid",
        "relevant_rank",
        "retrieved_pmids",
        "gold_in_top_5",
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


def create_context(top_documents, document_by_pmid):
    formatted_documents = []

    for index, retrieved_document in enumerate(
        top_documents,
        start=1,
    ):
        document_id = retrieved_document["pmid"]
        document = document_by_pmid[document_id]

        formatted_documents.append(
            f"[Document {index}]\n"
            f"{document['context']}"
        )

    return "\n\n".join(formatted_documents)

def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY nije pronađen.")

    if not model:
        raise ValueError("OPENAI_MODEL nije pronađen.")

    client = OpenAI(api_key=api_key)

    corpus = load_corpus()
    questions = load_questions()

    document_by_pmid = {
        document["pmid"]: document
        for document in corpus
    }

    document_frequencies = calculate_document_frequencies(corpus)

    average_document_length = (
        sum(len(document["tokens"]) for document in corpus)
        / len(corpus)
    )

    results, completed_ids = load_existing_results()

    print("Model:", model)
    print("Documents:", len(corpus))
    print("Questions:", len(questions))
    print("Already completed:", len(completed_ids))
    print()

    for index, item in enumerate(questions, start=1):
        question_id = item["question_id"]

        if question_id in completed_ids:
            print(f"[{index}/20] {question_id} already completed.")
            continue

        ranked_documents = retrieve(
            question=item["question"],
            corpus=corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        top_documents = ranked_documents[:TOP_K]

        relevant_pmid = str(item["pmid"])

        relevant_rank = next(
            (
                rank
                for rank, document in enumerate(
                    ranked_documents,
                    start=1,
                )
                if document["pmid"] == relevant_pmid
            ),
            None,
        )

        retrieved_pmids = [
            document["pmid"]
            for document in top_documents
        ]

        retrieved_context = create_context(
            top_documents,
            document_by_pmid,
        )

        prompt = f"""
Use only the biomedical documents provided below.

Answer the research question with exactly one word:
yes or no

Do not provide an explanation.

Question:
{item["question"]}

Retrieved biomedical documents:
{retrieved_context}
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

        result = {
            "question_id": question_id,
            "relevant_pmid": relevant_pmid,
            "relevant_rank": relevant_rank,
            "retrieved_pmids": "|".join(retrieved_pmids),
            "gold_in_top_5": relevant_pmid in retrieved_pmids,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer,
            "raw_output": raw_output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(latency, 3),
        }

        results.append(result)
        completed_ids.add(question_id)
        save_results(results)

        print(
            f"[{index}/{len(questions)}] {question_id}: "
            f"rank={relevant_rank}, "
            f"correct={correct_answer}, "
            f"predicted={predicted_answer}, "
            f"success={result['is_correct']}"
        )

    correct_count = sum(
        str(result["is_correct"]).lower() == "true"
        for result in results
    )

    accuracy = correct_count / len(results) if results else 0

    total_input_tokens = sum(
        int(result["input_tokens"])
        for result in results
    )

    total_output_tokens = sum(
        int(result["output_tokens"])
        for result in results
    )

    print()
    print("Completed questions:", len(results))
    print("Correct predictions:", correct_count)
    print(f"Clean RAG accuracy: {accuracy:.2%}")
    print("Input tokens:", total_input_tokens)
    print("Output tokens:", total_output_tokens)
    print("Results:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
