import csv
import json
from pathlib import Path

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
    PROJECT_ROOT / "results" / "main_bm25_retrieval.csv"
)

TOP_K = 5


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results):
    columns = [
        "question_id",
        "correct_answer",
        "relevant_pmid",
        "relevant_rank",
        "retrieved_at_1",
        "retrieved_at_5",
        "reciprocal_rank",
        "top_5_pmids",
        "top_5_scores",
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
    corpus = load_corpus()
    questions = load_questions()

    document_frequencies = calculate_document_frequencies(corpus)
    average_document_length = (
        sum(len(document["tokens"]) for document in corpus)
        / len(corpus)
    )

    print(f"Corpus documents: {len(corpus)}")
    print(f"Evaluation questions: {len(questions)}")
    print(
        "Average document length: "
        f"{average_document_length:.2f} tokens"
    )
    print()

    results = []

    for index, question in enumerate(questions, start=1):
        ranked_documents = retrieve(
            question=question["question"],
            corpus=corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        relevant_pmid = str(question["pmid"])
        relevant_rank = find_rank(
            ranked_documents,
            relevant_pmid,
        )
        top_documents = ranked_documents[:TOP_K]

        reciprocal_rank = (
            1 / relevant_rank
            if relevant_rank is not None
            else 0
        )

        results.append(
            {
                "question_id": question["question_id"],
                "correct_answer": question["correct_answer"],
                "relevant_pmid": relevant_pmid,
                "relevant_rank": relevant_rank,
                "retrieved_at_1": relevant_rank == 1,
                "retrieved_at_5": (
                    relevant_rank is not None
                    and relevant_rank <= TOP_K
                ),
                "reciprocal_rank": round(reciprocal_rank, 6),
                "top_5_pmids": "|".join(
                    document["pmid"]
                    for document in top_documents
                ),
                "top_5_scores": "|".join(
                    f'{document["score"]:.4f}'
                    for document in top_documents
                ),
            }
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question['question_id']}: "
            f"relevant document rank={relevant_rank}"
        )

    save_results(results)

    recall_at_1 = sum(
        result["retrieved_at_1"]
        for result in results
    ) / len(results)

    recall_at_5 = sum(
        result["retrieved_at_5"]
        for result in results
    ) / len(results)

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"]
        for result in results
    ) / len(results)

    print()
    print(f"Recall@1: {recall_at_1:.2%}")
    print(f"Recall@5: {recall_at_5:.2%}")
    print(f"MRR: {mean_reciprocal_rank:.4f}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
