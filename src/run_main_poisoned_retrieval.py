import csv
import json
from pathlib import Path

from run_bm25_retrieval import (
    calculate_document_frequencies,
    load_corpus,
    retrieve,
    tokenize,
)
from run_poisoned_retrieval import find_rank


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "main_questions_100.json"
)

POISON_PATH = (
    PROJECT_ROOT / "data" / "poisoned" / "main_poison_documents_100.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "main_poisoned_retrieval.csv"
)

TOP_K = 5
ATTACK_VARIANT = "passage_only"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(questions, poison_documents):
    question_ids = {
        question["question_id"]
        for question in questions
    }
    poison_question_ids = {
        poison["target_question_id"]
        for poison in poison_documents
    }

    missing_poisons = question_ids - poison_question_ids
    unexpected_poisons = poison_question_ids - question_ids

    if missing_poisons:
        raise ValueError(
            "Missing poison documents for: "
            + ", ".join(sorted(missing_poisons))
        )

    if unexpected_poisons:
        raise ValueError(
            "Unexpected poison documents for: "
            + ", ".join(sorted(unexpected_poisons))
        )


def build_poisoned_corpus(clean_corpus, poison_documents):
    poisoned_corpus = list(clean_corpus)

    for poison in poison_documents:
        poison_text = poison["generated_passage"]
        poisoned_corpus.append(
            {
                "pmid": poison["poison_id"],
                "context": poison_text,
                "tokens": tokenize(poison_text),
            }
        )

    return poisoned_corpus


def save_results(results):
    columns = [
        "attack_variant",
        "question_id",
        "correct_answer",
        "target_answer",
        "gold_pmid",
        "gold_rank",
        "target_poison_id",
        "target_poison_rank",
        "target_poison_at_1",
        "target_poison_at_5",
        "gold_at_1",
        "gold_at_5",
        "number_of_poisons_in_top_5",
        "top_5_document_ids",
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
    questions = load_json(QUESTIONS_PATH)
    poison_documents = load_json(POISON_PATH)
    clean_corpus = load_corpus()

    validate_inputs(questions, poison_documents)

    poisoned_corpus = build_poisoned_corpus(
        clean_corpus,
        poison_documents,
    )

    poison_by_question = {
        poison["target_question_id"]: poison
        for poison in poison_documents
    }

    document_frequencies = calculate_document_frequencies(
        poisoned_corpus
    )
    average_document_length = (
        sum(
            len(document["tokens"])
            for document in poisoned_corpus
        )
        / len(poisoned_corpus)
    )

    print(f"Attack variant: {ATTACK_VARIANT}")
    print(f"Clean documents: {len(clean_corpus)}")
    print(f"Poison documents: {len(poison_documents)}")
    print(f"Total documents: {len(poisoned_corpus)}")
    print(f"Evaluation questions: {len(questions)}")
    print()

    results = []

    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        target_poison = poison_by_question[question_id]

        ranked_documents = retrieve(
            question=question["question"],
            corpus=poisoned_corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        gold_pmid = str(question["pmid"])
        target_poison_id = target_poison["poison_id"]
        gold_rank = find_rank(ranked_documents, gold_pmid)
        target_poison_rank = find_rank(
            ranked_documents,
            target_poison_id,
        )

        top_documents = ranked_documents[:TOP_K]
        top_document_ids = [
            document["pmid"]
            for document in top_documents
        ]

        poisons_in_top_5 = [
            document_id
            for document_id in top_document_ids
            if document_id.startswith("poison_")
        ]

        result = {
            "attack_variant": ATTACK_VARIANT,
            "question_id": question_id,
            "correct_answer": question["correct_answer"],
            "target_answer": target_poison["target_answer"],
            "gold_pmid": gold_pmid,
            "gold_rank": gold_rank,
            "target_poison_id": target_poison_id,
            "target_poison_rank": target_poison_rank,
            "target_poison_at_1": target_poison_rank == 1,
            "target_poison_at_5": (
                target_poison_rank is not None
                and target_poison_rank <= TOP_K
            ),
            "gold_at_1": gold_rank == 1,
            "gold_at_5": (
                gold_rank is not None
                and gold_rank <= TOP_K
            ),
            "number_of_poisons_in_top_5": len(poisons_in_top_5),
            "top_5_document_ids": "|".join(top_document_ids),
            "top_5_scores": "|".join(
                f'{document["score"]:.4f}'
                for document in top_documents
            ),
        }

        results.append(result)

        print(
            f"[{index}/{len(questions)}] {question_id}: "
            f"poison rank={target_poison_rank}, "
            f"original rank={gold_rank}, "
            f"poisons in top 5={len(poisons_in_top_5)}"
        )

    save_results(results)

    poison_asr_at_1 = sum(
        result["target_poison_at_1"]
        for result in results
    ) / len(results)
    poison_asr_at_5 = sum(
        result["target_poison_at_5"]
        for result in results
    ) / len(results)
    gold_recall_at_1 = sum(
        result["gold_at_1"]
        for result in results
    ) / len(results)
    gold_recall_at_5 = sum(
        result["gold_at_5"]
        for result in results
    ) / len(results)
    average_poisons_in_top_5 = sum(
        result["number_of_poisons_in_top_5"]
        for result in results
    ) / len(results)

    print()
    print(f"Target poison Retrieval ASR@1: {poison_asr_at_1:.2%}")
    print(f"Target poison Retrieval ASR@5: {poison_asr_at_5:.2%}")
    print(f"Original document Recall@1: {gold_recall_at_1:.2%}")
    print(f"Original document Recall@5: {gold_recall_at_5:.2%}")
    print(
        "Average poison documents in top 5: "
        f"{average_poisons_in_top_5:.2f}"
    )
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
