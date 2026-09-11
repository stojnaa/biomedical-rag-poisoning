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

    print("PubMedQA downloaded successfully.")
    print("Location:", OUTPUT_PATH)
    print("Size:", len(response.content), "bytes")


if __name__ == "__main__":
    main()
