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
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

POISON_PATH = (
    PROJECT_ROOT / "data" / "poisoned" / "poison_documents.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "attack_ablation_retrieval.csv"
)

TOP_K = 5

ATTACK_VARIANTS = {
    "exact_query": "attack_text",
    "passage_only": "generated_passage",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_poisoned_corpus(clean_corpus, poison_documents, text_field):
    corpus = list(clean_corpus)

    for poison in poison_documents:
        poison_text = poison[text_field]
        corpus.append(
            {
                "pmid": poison["poison_id"],
                "context": poison_text,
                "tokens": tokenize(poison_text),
            }
        )

    return corpus


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


def evaluate_variant(
    variant_name,
    text_field,
    questions,
    poison_documents,
    clean_corpus,
):
    poisoned_corpus = build_poisoned_corpus(
        clean_corpus,
        poison_documents,
        text_field,
    )

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

    poison_by_question = {
        poison["target_question_id"]: poison
        for poison in poison_documents
    }

    results = []

    print(f"Attack variant: {variant_name}")

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

        results.append(
            {
                "attack_variant": variant_name,
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
                "top_5_document_ids": "|".join(
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
            f"[{index}/{len(questions)}] {question_id}: "
            f"poison rank={target_poison_rank}, "
            f"original rank={gold_rank}"
        )

    return results


def print_summary(results, variant_name):
    variant_results = [
        result
        for result in results
        if result["attack_variant"] == variant_name
    ]

    poison_asr_at_1 = sum(
        result["target_poison_at_1"]
        for result in variant_results
    ) / len(variant_results)

    poison_asr_at_5 = sum(
        result["target_poison_at_5"]
        for result in variant_results
    ) / len(variant_results)

    gold_recall_at_1 = sum(
        result["gold_at_1"]
        for result in variant_results
    ) / len(variant_results)

    gold_recall_at_5 = sum(
        result["gold_at_5"]
        for result in variant_results
    ) / len(variant_results)

    print()
    print(f"Summary for {variant_name}")
    print(f"Target poison Retrieval ASR@1: {poison_asr_at_1:.2%}")
    print(f"Target poison Retrieval ASR@5: {poison_asr_at_5:.2%}")
    print(f"Original document Recall@1: {gold_recall_at_1:.2%}")
    print(f"Original document Recall@5: {gold_recall_at_5:.2%}")


def main():
    questions = load_json(QUESTIONS_PATH)
    poison_documents = load_json(POISON_PATH)
    clean_corpus = load_corpus()

    print(f"Clean documents: {len(clean_corpus)}")
    print(f"Poison documents per variant: {len(poison_documents)}")
    print(f"Evaluation questions: {len(questions)}")
    print()

    all_results = []

    for variant_name, text_field in ATTACK_VARIANTS.items():
        variant_results = evaluate_variant(
            variant_name=variant_name,
            text_field=text_field,
            questions=questions,
            poison_documents=poison_documents,
            clean_corpus=clean_corpus,
        )

        all_results.extend(variant_results)
        print_summary(all_results, variant_name)
        print("-" * 60)

    save_results(all_results)

    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
