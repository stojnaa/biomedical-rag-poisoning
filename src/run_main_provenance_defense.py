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
    PROJECT_ROOT / "data" / "processed" / "main_questions_100.json"
)
POISON_PATH = (
    PROJECT_ROOT / "data" / "poisoned" / "main_poison_documents_100.json"
)
CLEAN_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "main_clean_rag.csv"
)
POISONED_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "main_poisoned_rag.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "main_provenance_defense_summary.csv"
)
DETAIL_PATH = (
    PROJECT_ROOT / "results" / "main_provenance_defense_retrieval.csv"
)

TOP_K = 5
DEFENSE_MODES = (
    "no_defense",
    "pmid_allowlist",
    "content_hash_verification",
)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv_by_question(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        row["question_id"]: row
        for row in rows
    }


def as_boolean(value):
    if isinstance(value, bool):
        return value

    return str(value).lower() == "true"


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def calculate_hash(text):
    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()


def create_candidates(clean_corpus, poison_documents):
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
                "claimed_pmid": poison["target_pmid"],
                "content": poison["generated_passage"],
                "is_poison": True,
            }
        )

    return candidates


def validate_document(document, canonical_hashes, mode):
    if mode == "no_defense":
        return True, "not_checked"

    claimed_pmid = document["claimed_pmid"]

    if claimed_pmid not in canonical_hashes:
        return False, "unknown_pmid"

    if mode == "pmid_allowlist":
        return True, "known_pmid"

    if mode == "content_hash_verification":
        submitted_hash = calculate_hash(document["content"])
        expected_hash = canonical_hashes[claimed_pmid]

        if submitted_hash != expected_hash:
            return False, "content_mismatch"

        return True, "verified_content"

    raise ValueError(f"Unsupported defense mode: {mode}")


def calculate_detection_metrics(accepted, rejected):
    true_positives = sum(
        document["is_poison"]
        for document in rejected
    )
    false_positives = sum(
        not document["is_poison"]
        for document in rejected
    )
    false_negatives = sum(
        document["is_poison"]
        for document in accepted
    )
    true_negatives = sum(
        not document["is_poison"]
        for document in accepted
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )
    false_positive_rate = (
        false_positives / (false_positives + true_negatives)
        if false_positives + true_negatives
        else 0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1_score,
        "false_positive_rate": false_positive_rate,
    }


def build_retrieval_corpus(accepted_documents):
    return [
        {
            "pmid": document["document_id"],
            "context": document["content"],
            "tokens": tokenize(document["content"]),
        }
        for document in accepted_documents
    ]


def evaluate_retrieval(
    mode,
    questions,
    poison_documents,
    accepted_documents,
):
    corpus = build_retrieval_corpus(accepted_documents)
    document_frequencies = calculate_document_frequencies(corpus)
    average_document_length = (
        sum(len(document["tokens"]) for document in corpus)
        / len(corpus)
    )

    poison_by_question = {
        poison["target_question_id"]: poison
        for poison in poison_documents
    }
    details = []

    for question in questions:
        question_id = question["question_id"]
        poison = poison_by_question[question_id]
        ranked_documents = retrieve(
            question=question["question"],
            corpus=corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

        gold_pmid = str(question["pmid"])
        target_poison_id = poison["poison_id"]
        gold_rank = find_rank(ranked_documents, gold_pmid)
        target_poison_rank = find_rank(
            ranked_documents,
            target_poison_id,
        )
        top_documents = ranked_documents[:TOP_K]

        details.append(
            {
                "defense_mode": mode,
                "question_id": question_id,
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
            }
        )

    return details


def calculate_end_to_end_metrics(
    mode,
    questions,
    clean_results,
    poisoned_results,
):
    poison_by_question = {
        row["question_id"]: row
        for row in poisoned_results.values()
    }

    if mode in ("no_defense", "pmid_allowlist"):
        selected_results = poisoned_results
        accuracy = sum(
            as_boolean(row["correct_after_attack"])
            for row in selected_results.values()
        ) / len(questions)
        raw_asr = sum(
            as_boolean(row["raw_attack_success"])
            for row in selected_results.values()
        ) / len(questions)
        eligible = sum(
            as_boolean(row["eligible_for_induced_attack"])
            for row in selected_results.values()
        )
        induced_successes = sum(
            as_boolean(row["induced_attack_success"])
            for row in selected_results.values()
        )
        induced_asr = induced_successes / eligible if eligible else 0

        return accuracy, raw_asr, induced_asr, induced_successes, eligible

    correct_predictions = 0
    target_predictions = 0

    for question in questions:
        question_id = question["question_id"]
        clean_prediction = clean_results[question_id]["predicted_answer"]
        target_answer = poison_by_question[question_id]["target_answer"]

        correct_predictions += (
            clean_prediction == question["correct_answer"]
        )
        target_predictions += clean_prediction == target_answer

    accuracy = correct_predictions / len(questions)
    raw_target_rate = target_predictions / len(questions)

    return accuracy, raw_target_rate, 0.0, 0, correct_predictions


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    questions = load_json(QUESTIONS_PATH)
    poison_documents = load_json(POISON_PATH)
    clean_results = load_csv_by_question(CLEAN_RESULTS_PATH)
    poisoned_results = load_csv_by_question(POISONED_RESULTS_PATH)
    clean_corpus = load_corpus()

    expected_question_ids = {
        question["question_id"]
        for question in questions
    }

    if set(clean_results) != expected_question_ids:
        raise ValueError("Clean RAG results are incomplete or mismatched.")

    if set(poisoned_results) != expected_question_ids:
        raise ValueError("Poisoned RAG results are incomplete or mismatched.")

    canonical_hashes = {
        document["pmid"]: calculate_hash(document["context"])
        for document in clean_corpus
    }
    candidates = create_candidates(clean_corpus, poison_documents)

    clean_accuracy = sum(
        as_boolean(row["is_correct"])
        for row in clean_results.values()
    ) / len(clean_results)

    summary_rows = []
    detail_rows = []

    print(f"Candidate documents: {len(candidates)}")
    print(f"Clean documents: {len(clean_corpus)}")
    print(f"Poison documents: {len(poison_documents)}")
    print(f"Clean RAG accuracy: {clean_accuracy:.2%}")
    print()

    for mode in DEFENSE_MODES:
        start_time = time.perf_counter()
        accepted = []
        rejected = []

        for document in candidates:
            is_valid, reason = validate_document(
                document,
                canonical_hashes,
                mode,
            )
            evaluated_document = {
                **document,
                "is_valid": is_valid,
                "reason": reason,
            }

            if is_valid:
                accepted.append(evaluated_document)
            else:
                rejected.append(evaluated_document)

        defense_latency = time.perf_counter() - start_time
        detection = calculate_detection_metrics(accepted, rejected)
        retrieval_details = evaluate_retrieval(
            mode,
            questions,
            poison_documents,
            accepted,
        )
        detail_rows.extend(retrieval_details)

        poison_asr_at_1 = sum(
            row["target_poison_at_1"]
            for row in retrieval_details
        ) / len(retrieval_details)
        poison_asr_at_5 = sum(
            row["target_poison_at_5"]
            for row in retrieval_details
        ) / len(retrieval_details)
        gold_recall_at_1 = sum(
            row["gold_at_1"]
            for row in retrieval_details
        ) / len(retrieval_details)
        gold_recall_at_5 = sum(
            row["gold_at_5"]
            for row in retrieval_details
        ) / len(retrieval_details)

        (
            defended_accuracy,
            raw_target_rate,
            induced_asr,
            induced_successes,
            eligible_questions,
        ) = calculate_end_to_end_metrics(
            mode,
            questions,
            clean_results,
            poisoned_results,
        )

        summary = {
            "defense_mode": mode,
            "accepted_documents": len(accepted),
            "rejected_documents": len(rejected),
            **detection,
            "poison_retrieval_asr_at_1": poison_asr_at_1,
            "poison_retrieval_asr_at_5": poison_asr_at_5,
            "gold_recall_at_1": gold_recall_at_1,
            "gold_recall_at_5": gold_recall_at_5,
            "end_to_end_accuracy": defended_accuracy,
            "utility_drop_from_clean": clean_accuracy - defended_accuracy,
            "raw_target_answer_rate": raw_target_rate,
            "attack_induced_asr": induced_asr,
            "induced_successes": induced_successes,
            "eligible_questions": eligible_questions,
            "defense_latency_seconds": round(defense_latency, 6),
        }
        summary_rows.append(summary)

        print(f"Defense mode: {mode}")
        print(f"Accepted documents: {len(accepted)}")
        print(f"Rejected documents: {len(rejected)}")
        print(f"Detection F1: {detection['detection_f1']:.2%}")
        print(f"False-positive rate: {detection['false_positive_rate']:.2%}")
        print(f"Poison Retrieval ASR@1: {poison_asr_at_1:.2%}")
        print(f"Poison Retrieval ASR@5: {poison_asr_at_5:.2%}")
        print(f"Original Recall@1: {gold_recall_at_1:.2%}")
        print(f"Original Recall@5: {gold_recall_at_5:.2%}")
        print(f"End-to-end accuracy: {defended_accuracy:.2%}")
        print(f"Utility drop from clean: {clean_accuracy - defended_accuracy:.2%}")
        print(f"Attack-induced ASR: {induced_asr:.2%}")
        print("-" * 60)

    save_csv(SUMMARY_PATH, summary_rows)
    save_csv(DETAIL_PATH, detail_rows)

    print()
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Retrieval details saved to: {DETAIL_PATH}")


if __name__ == "__main__":
    main()
