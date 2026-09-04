import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ori_pqal.json"
)

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "bm25_retrieval.csv"
)

TOP_K = 5
K1 = 1.5
B = 0.75


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus():
    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        raw_dataset = json.load(file)

    corpus = []

    for pmid, item in raw_dataset.items():
        context = " ".join(
            part.strip()
            for part in item.get("CONTEXTS", [])
        )

        if not context:
            continue

        corpus.append(
            {
                "pmid": str(pmid),
                "context": context,
                "tokens": tokenize(context),
            }
        )

    return corpus


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_document_frequencies(corpus):
    document_frequencies = Counter()

    for document in corpus:
        unique_terms = set(document["tokens"])
        document_frequencies.update(unique_terms)

    return document_frequencies


def calculate_bm25_score(
    query_tokens,
    document_tokens,
    document_frequencies,
    corpus_size,
    average_document_length,
):
    term_frequencies = Counter(document_tokens)
    document_length = len(document_tokens)

    score = 0.0

    for term in set(query_tokens):
        if term not in term_frequencies:
            continue

        document_frequency = document_frequencies[term]

        inverse_document_frequency = math.log(
            1
            + (
                corpus_size
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

        term_frequency = term_frequencies[term]

        numerator = term_frequency * (K1 + 1)

        denominator = (
            term_frequency
            + K1
            * (
                1
                - B
                + B
                * document_length
                / average_document_length
            )
        )

        score += (
            inverse_document_frequency
            * numerator
            / denominator
        )

    return score


def retrieve(
    question,
    corpus,
    document_frequencies,
    average_document_length,
):
    query_tokens = tokenize(question)
    scored_documents = []

    for document in corpus:
        score = calculate_bm25_score(
            query_tokens=query_tokens,
            document_tokens=document["tokens"],
            document_frequencies=document_frequencies,
            corpus_size=len(corpus),
            average_document_length=average_document_length,
        )

        scored_documents.append(
            {
                "pmid": document["pmid"],
                "score": score,
            }
        )

    scored_documents.sort(
        key=lambda item: (-item["score"], item["pmid"])
    )

    return scored_documents


def save_results(results):
    columns = [
        "question_id",
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

    print("Broj dokumenata u bazi:", len(corpus))
    print(
        "Prosečna dužina dokumenta:",
        round(average_document_length, 2),
        "tokena",
    )
    print()

    results = []

    for index, item in enumerate(questions, start=1):
        ranked_documents = retrieve(
            question=item["question"],
            corpus=corpus,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
        )

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

        top_documents = ranked_documents[:TOP_K]
        top_pmids = [
            document["pmid"]
            for document in top_documents
        ]

        retrieved_at_1 = relevant_rank == 1
        retrieved_at_5 = (
            relevant_rank is not None
            and relevant_rank <= TOP_K
        )

        reciprocal_rank = (
            1 / relevant_rank
            if relevant_rank is not None
            else 0
        )

        results.append(
            {
                "question_id": item["question_id"],
                "relevant_pmid": relevant_pmid,
                "relevant_rank": relevant_rank,
                "retrieved_at_1": retrieved_at_1,
                "retrieved_at_5": retrieved_at_5,
                "reciprocal_rank": round(
                    reciprocal_rank,
                    6,
                ),
                "top_5_pmids": "|".join(top_pmids),
                "top_5_scores": "|".join(
                    f'{document["score"]:.4f}'
                    for document in top_documents
                ),
            }
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{item['question_id']}: "
            f"relevantni dokument je na mestu "
            f"{relevant_rank}"
        )

    save_results(results)

    recall_at_1 = sum(
        result["retrieved_at_1"] for result in results
    ) / len(results)

    recall_at_5 = sum(
        result["retrieved_at_5"] for result in results
    ) / len(results)

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"] for result in results
    ) / len(results)

    print()
    print(f"Recall@1: {recall_at_1:.2%}")
    print(f"Recall@5: {recall_at_5:.2%}")
    print(f"MRR: {mean_reciprocal_rank:.4f}")
    print("Rezultati:", OUTPUT_PATH)


if __name__ == "__main__":
    main()