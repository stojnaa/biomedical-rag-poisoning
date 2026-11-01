"""Dense-retriever robustness check for the main 100-question experiment.

The dense retriever is evaluated on the same corpora as BM25:

- clean corpus: 1,000 PubMedQA PQA-L documents
- attacked corpus: 1,000 clean documents + 100 poison passages

Outputs:
- results/main_dense_retrieval.csv
- results/main_dense_retrieval_summary.csv
- results/main_dense_retrieval_metadata.json

Run from the repository root:

    python src/dense_retriever_check.py
"""

import csv
import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from run_bm25_retrieval import load_corpus


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "main_questions_100.json"
)

POISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "poisoned"
    / "main_poison_documents_100.json"
)

PER_QUESTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "main_dense_retrieval.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "main_dense_retrieval_summary.csv"
)

METADATA_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "main_dense_retrieval_metadata.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = 5

EXPECTED_QUESTIONS = 100
EXPECTED_CLEAN_DOCUMENTS = 1_000
EXPECTED_POISON_DOCUMENTS = 100

Z_95 = 1.959963984540054


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_inputs(
    questions,
    poison_documents,
    clean_corpus,
):
    if len(questions) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions, "
            f"got {len(questions)}."
        )

    if len(clean_corpus) != EXPECTED_CLEAN_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_CLEAN_DOCUMENTS} clean "
            f"documents, got {len(clean_corpus)}. "
            "Run src/download_pubmedqa.py first."
        )

    if len(poison_documents) != EXPECTED_POISON_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_POISON_DOCUMENTS} poison "
            f"documents, got {len(poison_documents)}."
        )

    question_ids = [
        str(question["question_id"])
        for question in questions
    ]

    question_pmids = [
        str(question["pmid"])
        for question in questions
    ]

    poison_ids = [
        str(poison["poison_id"])
        for poison in poison_documents
    ]

    poison_question_ids = [
        str(poison["target_question_id"])
        for poison in poison_documents
    ]

    clean_document_ids = [
        str(document["pmid"])
        for document in clean_corpus
    ]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError(
            "Question IDs must be unique."
        )

    if len(set(question_pmids)) != len(question_pmids):
        raise ValueError(
            "Evaluation PMIDs must be unique."
        )

    if len(set(poison_ids)) != len(poison_ids):
        raise ValueError(
            "Poison IDs must be unique."
        )

    if (
        len(set(poison_question_ids))
        != len(poison_question_ids)
    ):
        raise ValueError(
            "Each evaluation question must have "
            "exactly one poison passage."
        )

    if (
        len(set(clean_document_ids))
        != len(clean_document_ids)
    ):
        raise ValueError(
            "Clean document IDs must be unique."
        )

    if set(question_ids) != set(poison_question_ids):
        missing_poisons = sorted(
            set(question_ids)
            - set(poison_question_ids)
        )

        unexpected_poisons = sorted(
            set(poison_question_ids)
            - set(question_ids)
        )

        raise ValueError(
            "Question/poison alignment failed. "
            f"Missing poisons: {missing_poisons}; "
            f"unexpected poisons: {unexpected_poisons}."
        )

    missing_gold_documents = sorted(
        set(question_pmids)
        - set(clean_document_ids)
    )

    if missing_gold_documents:
        raise ValueError(
            "Gold documents missing from clean corpus: "
            + ", ".join(missing_gold_documents)
        )

    overlapping_ids = (
        set(clean_document_ids)
        .intersection(poison_ids)
    )

    if overlapping_ids:
        raise ValueError(
            "Poison IDs overlap clean document IDs: "
            + ", ".join(sorted(overlapping_ids))
        )


def build_attacked_corpus(
    clean_corpus,
    poison_documents,
):
    attacked_corpus = [
        {
            "document_id": str(document["pmid"]),
            "document_type": "clean",
            "context": document["context"],
        }
        for document in clean_corpus
    ]

    attacked_corpus.extend(
        {
            "document_id": str(poison["poison_id"]),
            "document_type": "poison",
            "context": poison["generated_passage"],
        }
        for poison in poison_documents
    )

    expected_total = (
        EXPECTED_CLEAN_DOCUMENTS
        + EXPECTED_POISON_DOCUMENTS
    )

    if len(attacked_corpus) != expected_total:
        raise ValueError(
            f"Expected {expected_total} attacked-corpus "
            f"documents, got {len(attacked_corpus)}."
        )

    return attacked_corpus


def inverse_ranks(scores):
    """Return ranked indices and one-based ranks.

    Stable sorting gives deterministic ordering if two
    documents have the same similarity score.
    """

    ranked_indices = np.argsort(
        -scores,
        kind="stable",
    )

    ranks = np.empty(
        len(ranked_indices),
        dtype=np.int64,
    )

    ranks[ranked_indices] = np.arange(
        1,
        len(ranked_indices) + 1,
    )

    return ranked_indices, ranks


def wilson_interval(successes, total):
    if total == 0:
        return 0.0, 0.0

    proportion = successes / total

    denominator = (
        1
        + Z_95**2 / total
    )

    center = (
        proportion
        + Z_95**2 / (2 * total)
    ) / denominator

    margin = (
        Z_95
        * math.sqrt(
            proportion
            * (1 - proportion)
            / total
            + Z_95**2
            / (4 * total**2)
        )
        / denominator
    )

    return (
        center - margin,
        center + margin,
    )


def create_proportion_metric(
    name,
    successes,
    total,
    condition,
):
    lower, upper = wilson_interval(
        successes,
        total,
    )

    return {
        "metric": name,
        "condition": condition,
        "successes": successes,
        "total": total,
        "value": (
            successes / total
            if total
            else 0.0
        ),
        "ci_95_lower": lower,
        "ci_95_upper": upper,
    }


def create_scalar_metric(
    name,
    value,
    total,
    condition,
):
    return {
        "metric": name,
        "condition": condition,
        "successes": "",
        "total": total,
        "value": value,
        "ci_95_lower": "",
        "ci_95_upper": "",
    }


def save_csv(
    path,
    rows,
    columns,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def get_sentence_transformers_version():
    try:
        return version(
            "sentence-transformers"
        )
    except PackageNotFoundError:
        return "unknown"


def main():
    questions = load_json(
        QUESTIONS_PATH
    )

    poison_documents = load_json(
        POISON_PATH
    )

    try:
        clean_corpus = load_corpus()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}\n"
            "Run 'python src/download_pubmedqa.py' first."
        ) from error

    validate_inputs(
        questions,
        poison_documents,
        clean_corpus,
    )

    attacked_corpus = build_attacked_corpus(
        clean_corpus,
        poison_documents,
    )

    clean_document_count = len(
        clean_corpus
    )

    attacked_document_ids = [
        document["document_id"]
        for document in attacked_corpus
    ]

    clean_index_by_id = {
        document_id: index
        for index, document_id
        in enumerate(
            attacked_document_ids[
                :clean_document_count
            ]
        )
    }

    attacked_index_by_id = {
        document_id: index
        for index, document_id
        in enumerate(attacked_document_ids)
    }

    poison_by_question = {
        str(poison["target_question_id"]): poison
        for poison in poison_documents
    }

    poison_id_set = {
        str(poison["poison_id"])
        for poison in poison_documents
    }

    print(
        f"Loading dense retriever: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    document_texts = [
        document["context"]
        for document in attacked_corpus
    ]

    question_texts = [
        question["question"]
        for question in questions
    ]

    print(
        "Embedding 1,100 documents "
        "(1,000 clean + 100 poison)..."
    )

    document_embeddings = model.encode(
        document_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(
        "Embedding 100 evaluation questions..."
    )

    question_embeddings = model.encode(
        question_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Embeddings are normalized, so matrix
    # multiplication gives cosine similarity.
    attacked_score_matrix = (
        question_embeddings
        @ document_embeddings.T
    )

    # The first 1,000 rows of the document matrix
    # correspond to the clean PubMedQA corpus.
    clean_score_matrix = (
        attacked_score_matrix[
            :,
            :clean_document_count,
        ]
    )

    results = []

    clean_reciprocal_ranks = []
    attacked_reciprocal_ranks = []

    for question_index, question in enumerate(
        questions
    ):
        question_id = str(
            question["question_id"]
        )

        gold_pmid = str(
            question["pmid"]
        )

        target_poison = (
            poison_by_question[
                question_id
            ]
        )

        target_poison_id = str(
            target_poison["poison_id"]
        )

        (
            _,
            clean_ranks,
        ) = inverse_ranks(
            clean_score_matrix[
                question_index
            ]
        )

        (
            attacked_ranked_indices,
            attacked_ranks,
        ) = inverse_ranks(
            attacked_score_matrix[
                question_index
            ]
        )

        clean_gold_rank = int(
            clean_ranks[
                clean_index_by_id[
                    gold_pmid
                ]
            ]
        )

        attacked_gold_rank = int(
            attacked_ranks[
                attacked_index_by_id[
                    gold_pmid
                ]
            ]
        )

        target_poison_rank = int(
            attacked_ranks[
                attacked_index_by_id[
                    target_poison_id
                ]
            ]
        )

        clean_reciprocal_ranks.append(
            1.0 / clean_gold_rank
        )

        attacked_reciprocal_ranks.append(
            1.0 / attacked_gold_rank
        )

        top_indices = (
            attacked_ranked_indices[
                :TOP_K
            ]
        )

        top_document_ids = [
            attacked_document_ids[index]
            for index in top_indices
        ]

        top_scores = [
            float(
                attacked_score_matrix[
                    question_index,
                    index,
                ]
            )
            for index in top_indices
        ]

        result = {
            "retriever": MODEL_NAME,
            "question_id": question_id,
            "correct_answer": (
                question["correct_answer"]
            ),
            "target_answer": (
                target_poison[
                    "target_answer"
                ]
            ),
            "gold_pmid": gold_pmid,
            "clean_gold_rank": (
                clean_gold_rank
            ),
            "attacked_gold_rank": (
                attacked_gold_rank
            ),
            "target_poison_id": (
                target_poison_id
            ),
            "target_poison_rank": (
                target_poison_rank
            ),
            "target_poison_at_1": (
                target_poison_rank <= 1
            ),
            "target_poison_at_2": (
                target_poison_rank <= 2
            ),
            "target_poison_at_3": (
                target_poison_rank <= 3
            ),
            "target_poison_at_4": (
                target_poison_rank <= 4
            ),
            "target_poison_at_5": (
                target_poison_rank <= 5
            ),
            "clean_gold_at_1": (
                clean_gold_rank <= 1
            ),
            "clean_gold_at_5": (
                clean_gold_rank <= 5
            ),
            "attacked_gold_at_1": (
                attacked_gold_rank <= 1
            ),
            "attacked_gold_at_2": (
                attacked_gold_rank <= 2
            ),
            "attacked_gold_at_3": (
                attacked_gold_rank <= 3
            ),
            "attacked_gold_at_4": (
                attacked_gold_rank <= 4
            ),
            "attacked_gold_at_5": (
                attacked_gold_rank <= 5
            ),
            "number_of_poisons_in_top_5": sum(
                document_id
                in poison_id_set
                for document_id
                in top_document_ids
            ),
            "top_5_document_ids": "|".join(
                top_document_ids
            ),
            "top_5_scores": "|".join(
                f"{score:.8f}"
                for score in top_scores
            ),
        }

        results.append(result)

        print(
            f"[{question_index + 1}/100] "
            f"{question_id}: "
            f"clean gold rank="
            f"{clean_gold_rank}, "
            f"attacked gold rank="
            f"{attacked_gold_rank}, "
            f"target poison rank="
            f"{target_poison_rank}"
        )

    per_question_columns = [
        "retriever",
        "question_id",
        "correct_answer",
        "target_answer",
        "gold_pmid",
        "clean_gold_rank",
        "attacked_gold_rank",
        "target_poison_id",
        "target_poison_rank",
        "target_poison_at_1",
        "target_poison_at_2",
        "target_poison_at_3",
        "target_poison_at_4",
        "target_poison_at_5",
        "clean_gold_at_1",
        "clean_gold_at_5",
        "attacked_gold_at_1",
        "attacked_gold_at_2",
        "attacked_gold_at_3",
        "attacked_gold_at_4",
        "attacked_gold_at_5",
        "number_of_poisons_in_top_5",
        "top_5_document_ids",
        "top_5_scores",
    ]

    save_csv(
        PER_QUESTION_OUTPUT_PATH,
        results,
        per_question_columns,
    )

    total = len(results)
    summary = []

    for k in range(
        1,
        TOP_K + 1,
    ):
        field = (
            f"target_poison_at_{k}"
        )

        successes = sum(
            result[field]
            for result in results
        )

        summary.append(
            create_proportion_metric(
                name=(
                    "Poison Retrieval "
                    f"ASR@{k}"
                ),
                successes=successes,
                total=total,
                condition="attacked_1100",
            )
        )

    clean_recall_at_1 = sum(
        result["clean_gold_at_1"]
        for result in results
    )

    clean_recall_at_5 = sum(
        result["clean_gold_at_5"]
        for result in results
    )

    summary.extend(
        [
            create_proportion_metric(
                name="Gold Recall@1",
                successes=clean_recall_at_1,
                total=total,
                condition="clean_1000",
            ),
            create_proportion_metric(
                name="Gold Recall@5",
                successes=clean_recall_at_5,
                total=total,
                condition="clean_1000",
            ),
            create_scalar_metric(
                name="Gold MRR",
                value=float(
                    np.mean(
                        clean_reciprocal_ranks
                    )
                ),
                total=total,
                condition="clean_1000",
            ),
        ]
    )

    for k in range(
        1,
        TOP_K + 1,
    ):
        field = (
            f"attacked_gold_at_{k}"
        )

        successes = sum(
            result[field]
            for result in results
        )

        summary.append(
            create_proportion_metric(
                name=f"Gold Recall@{k}",
                successes=successes,
                total=total,
                condition="attacked_1100",
            )
        )

    summary.append(
        create_scalar_metric(
            name="Gold MRR",
            value=float(
                np.mean(
                    attacked_reciprocal_ranks
                )
            ),
            total=total,
            condition="attacked_1100",
        )
    )

    summary_columns = [
        "metric",
        "condition",
        "successes",
        "total",
        "value",
        "ci_95_lower",
        "ci_95_upper",
    ]

    save_csv(
        SUMMARY_OUTPUT_PATH,
        summary,
        summary_columns,
    )

    metadata = {
        "retriever": MODEL_NAME,
        "sentence_transformers_version": (
            get_sentence_transformers_version()
        ),
        "device": str(model.device),
        "embedding_dimension": (
            model
            .get_sentence_embedding_dimension()
        ),
        "max_sequence_length": (
            model.max_seq_length
        ),
        "batch_size": BATCH_SIZE,
        "similarity": (
            "cosine via dot product "
            "of normalized embeddings"
        ),
        "tie_breaking": (
            "stable sort using "
            "fixed corpus order"
        ),
        "evaluation_questions": (
            len(questions)
        ),
        "clean_documents": (
            clean_document_count
        ),
        "poison_documents": (
            len(poison_documents)
        ),
        "attacked_documents": (
            len(attacked_corpus)
        ),
    }

    with METADATA_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== DENSE RETRIEVER SUMMARY ==="
    )

    for result in summary:
        if result["ci_95_lower"] == "":
            print(
                f'{result["condition"]} | '
                f'{result["metric"]}: '
                f'{result["value"]:.4f}'
            )
        else:
            print(
                f'{result["condition"]} | '
                f'{result["metric"]}: '
                f'{result["value"]:.2%} '
                f'({result["successes"]}/'
                f'{result["total"]}), '
                f'95% CI '
                f'[{result["ci_95_lower"]:.2%}, '
                f'{result["ci_95_upper"]:.2%}]'
            )

    print(
        "\nPer-question results: "
        f"{PER_QUESTION_OUTPUT_PATH}"
    )

    print(
        "Summary results: "
        f"{SUMMARY_OUTPUT_PATH}"
    )

    print(
        "Metadata: "
        f"{METADATA_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()