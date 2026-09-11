import csv
import hashlib
import json
import re
import time
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
    PROJECT_ROOT / "results" / "provenance_defense.csv"
)

TOP_K = 5


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def calculate_hash(text):
    normalized_text = normalize_text(text)

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def create_candidate_documents(
    clean_corpus,
    poison_documents,
):
    candidates = []

    for document in clean_corpus:
        candidates.append(
            {
                "document_id": document["pmid"],
                "claimed_pmid": document["pmid"],
                "content": document["context"],
                "is_poison": False,
            }
        )

    for poison in poison_documents:
        candidates.append(
            {
                "document_id": poison["poison_id"],

                # Napadač tvrdi da tekst pripada
                # originalnom PubMed radu.
                "claimed_pmid": poison["target_pmid"],

                "content": poison["attack_text"],
                "is_poison": True,
            }
        )

    return candidates


def validate_document(
    document,
    canonical_hashes,
):
    claimed_pmid = document["claimed_pmid"]

    if claimed_pmid not in canonical_hashes:
        return False, "unknown_pmid"

    submitted_hash = calculate_hash(
        document["content"]
    )

    expected_hash = canonical_hashes[
        claimed_pmid
    ]

    if submitted_hash != expected_hash:
        return False, "content_mismatch"

    return True, "verified"


def save_results(results):
    columns = [
        "question_id",
        "gold_pmid",
        "gold_rank_after_defense",
        "gold_at_1_after_defense",
        "gold_at_5_after_defense",
        "target_poison_id",
        "target_poison_present_after_defense",
        "top_5_document_ids",
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

    canonical_hashes = {
        document["pmid"]: calculate_hash(
            document["context"]
        )
        for document in clean_corpus
    }

    candidates = create_candidate_documents(
        clean_corpus,
        poison_documents,
    )

    start_time = time.perf_counter()

    accepted_documents = []
    rejected_documents = []

    for document in candidates:
        is_valid, reason = validate_document(
            document,
            canonical_hashes,
        )

        evaluated_document = {
            **document,
            "is_valid": is_valid,
            "reason": reason,
        }

        if is_valid:
            accepted_documents.append(
                evaluated_document
            )
        else:
            rejected_documents.append(
                evaluated_document
            )

    defense_latency = time.perf_counter() - start_time

    true_positives = sum(
        document["is_poison"]
        for document in rejected_documents
    )

    false_positives = sum(
        not document["is_poison"]
        for document in rejected_documents
    )

    false_negatives = sum(
        document["is_poison"]
        for document in accepted_documents
    )

    true_negatives = sum(
        not document["is_poison"]
        for document in accepted_documents
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives
        else 0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0
    )

    f1_score = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    false_positive_rate = (
        false_positives
        / (false_positives + true_negatives)
        if false_positives + true_negatives
        else 0
    )

    filtered_corpus = [
        {
            "pmid": document["document_id"],
            "context": document["content"],
            "tokens": tokenize(
                document["content"]
            ),
        }
        for document in accepted_documents
    ]

    document_frequencies = calculate_document_frequencies(
        filtered_corpus
    )

    average_document_length = (
        sum(
            len(document["tokens"])
            for document in filtered_corpus
        )
        / len(filtered_corpus)
    )

    poison_by_question = {
        poison["target_question_id"]: poison
        for poison in poison_documents
    }

    results = []

    for index, question in enumerate(questions, start=1):
        ranked_documents = retrieve(
            question=question["question"],
            corpus=filtered_corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        gold_pmid = str(question["pmid"])

        poison_id = poison_by_question[
            question["question_id"]
        ]["poison_id"]

        gold_rank = find_rank(
            ranked_documents,
            gold_pmid,
        )

        top_document_ids = [
            document["pmid"]
            for document in ranked_documents[:TOP_K]
        ]

        poison_present = any(
            document["pmid"] == poison_id
            for document in ranked_documents
        )

        results.append(
            {
                "question_id": question["question_id"],
                "gold_pmid": gold_pmid,
                "gold_rank_after_defense": gold_rank,
                "gold_at_1_after_defense": (
                    gold_rank == 1
                ),
                "gold_at_5_after_defense": (
                    gold_rank is not None
                    and gold_rank <= TOP_K
                ),
                "target_poison_id": poison_id,
                "target_poison_present_after_defense": (
                    poison_present
                ),
                "top_5_document_ids": "|".join(
                    top_document_ids
                ),
            }
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question['question_id']}: "
            f"original rank={gold_rank}, "
            f"poison present={poison_present}"
        )

    save_results(results)

    print()
    print("Candidate documents:", len(candidates))
    print("Accepted documents:", len(accepted_documents))
    print("Rejected documents:", len(rejected_documents))
    print()
    print("TP - detected poisons:", true_positives)
    print("FP - rejected clean documents:", false_positives)
    print("FN - undetected poisons:", false_negatives)
    print("TN - accepted clean documents:", true_negatives)
    print()
    print(f"Defense precision: {precision:.2%}")
    print(f"Defense recall: {recall:.2%}")
    print(f"Defense F1: {f1_score:.2%}")
    print(
        f"False-positive rate: "
        f"{false_positive_rate:.2%}"
    )
    print(
        f"Defense latency: "
        f"{defense_latency:.6f} seconds"
    )
    print("Filtered corpus:", len(filtered_corpus))
    print("Results:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
