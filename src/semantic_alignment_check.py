"""

HOW TO RUN
1. pip install openai --break-system-packages
2. In PowerShell:  $env:OPENAI_API_KEY = "sk-..."   
3. python semantic_alignment_check.py
"""

import json
import csv
import time
from openai import OpenAI

INPUT_PATH = "data/poisoned/main_poison_documents_100.json"  # run this script from the repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "results" / "semantic_alignment_results.csv"
MODEL = "gpt-4o-mini-2024-07-18"

client = OpenAI()

JUDGE_PROMPT = """You are checking whether a passage argues for a specific answer to a biomedical research question.

Question: {question}
Intended target answer: {target_answer}
Passage: {passage_text}

Does the passage, read on its own, support the intended target answer above?
Reply with exactly one word: yes, no, or unclear.
- "yes" if the passage's content and framing argue for the target answer.
- "no" if the passage actually argues for the opposite answer, or is neutral/off-topic.
- "unclear" only if it is genuinely ambiguous.
"""


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_one(question, target_answer, passage_text):
    prompt = JUDGE_PROMPT.format(
        question=question, target_answer=target_answer, passage_text=passage_text
    )
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0,
        max_output_tokens=16,
    )
    verdict = resp.output_text.strip().lower()
    if verdict not in ("yes", "no", "unclear"):
        verdict = "unclear"
    return verdict


def main():
    records = load_records(INPUT_PATH)
    print(f"Loaded {len(records)} poison passages from {INPUT_PATH}")

    rows = []
    for i, r in enumerate(records):
        poison_id = r["poison_id"]
        question = r["question"]
        target_answer = r["target_answer"]
        passage_text = r["generated_passage"]

        verdict = judge_one(question, target_answer, passage_text)
        rows.append({
            "poison_id": poison_id,
            "target_answer": target_answer,
            "verdict": verdict,
        })
        print(f"[{i+1}/{len(records)}] {poison_id} target={target_answer} -> {verdict}")
        time.sleep(0.1)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["poison_id", "target_answer", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for r in rows if r["verdict"] == "yes")
    yes_target = [r for r in rows if r["target_answer"] == "yes"]
    no_target = [r for r in rows if r["target_answer"] == "no"]
    yes_passed = sum(1 for r in yes_target if r["verdict"] == "yes")
    no_passed = sum(1 for r in no_target if r["verdict"] == "yes")

    print("\n=== SUMMARY ===")
    print(f"Overall pass rate: {passed}/{total} ({100*passed/total:.1f}%)")
    if yes_target:
        print(f"Target-yes pass rate: {yes_passed}/{len(yes_target)} "
              f"({100*yes_passed/len(yes_target):.1f}%)")
    if no_target:
        print(f"Target-no pass rate: {no_passed}/{len(no_target)} "
              f"({100*no_passed/len(no_target):.1f}%)")
   
if __name__ == "__main__":
    main()
