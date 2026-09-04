from pathlib import Path

import requests


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "pubmedqa/pubmedqa/master/data/ori_pqal.json"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "ori_pqal.json"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    OUTPUT_PATH.write_bytes(response.content)

    print("PubMedQA je uspešno preuzet.")
    print("Lokacija:", OUTPUT_PATH)
    print("Veličina:", len(response.content), "bajtova")


if __name__ == "__main__":
    main()