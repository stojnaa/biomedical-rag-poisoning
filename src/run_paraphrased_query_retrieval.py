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

PARAPHRASES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_query_paraphrases.json"
)

POISON_PATH = (
    PROJECT_ROOT / "data" / "poisoned" / "poison_documents.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "paraphrased_query_retrieval.csv"
)

TOP_K = 5
ATTACK_VARIANT = "passage_only_paraphrased_query"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_poisoned_corpus(clean_corpus, poison_documents):
    corpus = list(clean_corpus)

    for poison in poison_documents:
        poison_text = poison["generated_passage"]
        corpus.append(
            {
                "pmid": poison["poison_id"],
                "context": poison_text,
                "tokens": tokenize(poison_text),
            }
        )

    return corpus


def validate_input(paraphrases):
    unvalidated_ids = [
        item["question_id"]
        for item in paraphrases
        if not item.get("manually_validated", False)
    ]

    if unvalidated_ids:
        raise ValueError(
            "Paraphrases must be validated before evaluation. "
            "Unvalidated question IDs: "
            + ", ".join(unvalidated_ids)
        )


def save_results(results):
    columns = [
        "attack_variant",
        "question_id",
        "original_question",
        "paraphrased_question",
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
    paraphrases = load_json(PARAPHRASES_PATH)
    poison_documents = load_json(POISON_PATH)
    clean_corpus = load_corpus()

    validate_input(paraphrases)

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
    print(f"Evaluation questions: {len(paraphrases)}")
    print()

    results = []

    for index, item in enumerate(paraphrases, start=1):
        question_id = item["question_id"]
        poison = poison_by_question[question_id]

        ranked_documents = retrieve(
            question=item["paraphrased_question"],
            corpus=poisoned_corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        gold_pmid = str(item["pmid"])
        target_poison_id = poison["poison_id"]

        gold_rank = find_rank(ranked_documents, gold_pmid)
        target_poison_rank = find_rank(
            ranked_documents,
            target_poison_id,
        )

        top_documents = ranked_documents[:TOP_K]

        result = {
            "attack_variant": ATTACK_VARIANT,
            "question_id": question_id,
            "original_question": item["original_question"],
            "paraphrased_question": item["paraphrased_question"],
            "correct_answer": item["correct_answer"],
            "target_answer": poison["target_answer"],
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
            "top_5_document_ids": "|".join(
                document["pmid"]
                for document in top_documents
            ),
            "top_5_scores": "|".join(
                f'{document["score"]:.4f}'
                for document in top_documents
            ),
        }

        results.append(result)

        print(
            f"[{index}/{len(paraphrases)}] {question_id}: "
            f"poison rank={target_poison_rank}, "
            f"original rank={gold_rank}"
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

    print()
    print(f"Target poison Retrieval ASR@1: {poison_asr_at_1:.2%}")
    print(f"Target poison Retrieval ASR@5: {poison_asr_at_5:.2%}")
    print(f"Original document Recall@1: {gold_recall_at_1:.2%}")
    print(f"Original document Recall@5: {gold_recall_at_5:.2%}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
