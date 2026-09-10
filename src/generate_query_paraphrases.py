import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pilot_query_paraphrases.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_existing_paraphrases():
    if not OUTPUT_PATH.exists():
        return []

    return load_json(OUTPUT_PATH)


def save_paraphrases(paraphrases):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            paraphrases,
            file,
            ensure_ascii=False,
            indent=2,
        )


def clean_paraphrase(text):
    cleaned = text.strip().strip('"').strip("'")

    if not cleaned.endswith("?"):
        cleaned += "?"

    return cleaned


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
    paraphrases = load_existing_paraphrases()

    completed_ids = {
        item["question_id"]
        for item in paraphrases
    }

    print(f"Model: {model}")
    print(f"Evaluation questions: {len(questions)}")
    print(f"Already generated: {len(completed_ids)}")
    print()

    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]

        if question_id in completed_ids:
            print(
                f"[{index}/{len(questions)}] "
                f"{question_id} already exists."
            )
            continue

        prompt = f"""
Paraphrase the biomedical research question below.

Requirements:
- Preserve the exact scientific meaning.
- Preserve all biomedical entities, interventions, comparisons,
  populations, and outcomes.
- Keep it answerable with yes or no.
- Do not answer the question.
- Do not add or remove scientific claims.
- Use different wording and sentence structure where possible.
- Return only one paraphrased question.

Original question:
{question["question"]}
""".strip()

        start_time = time.perf_counter()

        response = client.responses.create(
            model=model,
            temperature=0,
            max_output_tokens=100,
            input=prompt,
        )

        latency = time.perf_counter() - start_time
        paraphrased_question = clean_paraphrase(
            response.output_text
        )

        item = {
            "question_id": question_id,
            "pmid": str(question["pmid"]),
            "correct_answer": question["correct_answer"],
            "original_question": question["question"],
            "paraphrased_question": paraphrased_question,
            "manually_validated": False,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(latency, 3),
        }

        paraphrases.append(item)
        completed_ids.add(question_id)
        save_paraphrases(paraphrases)

        print(f"[{index}/{len(questions)}] {question_id}")
        print(f"Original:   {question['question']}")
        print(f"Paraphrase: {paraphrased_question}")
        print()

    total_input_tokens = sum(
        int(item["input_tokens"])
        for item in paraphrases
    )

    total_output_tokens = sum(
        int(item["output_tokens"])
        for item in paraphrases
    )

    print(f"Generated paraphrases: {len(paraphrases)}")
    print(f"Input tokens: {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
