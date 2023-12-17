import argparse
import os

import requests
from clint.textui import progress, prompt

# add your datasets here
targets = [
    ("parking.zip", "https://rpg.ifi.uzh.ch/docs/teaching/2023/parking.zip"),
    ("kitti05.zip", "https://rpg.ifi.uzh.ch/docs/teaching/2023/kitti05.zip"),
    (
        "malaga-urban-dataset-extract-07.zip",
        "https://rpg.ifi.uzh.ch/docs/teaching/2023/malaga-urban-dataset-extract-07.zip",
    ),
    (
        "woko.zip",
        "https://nextcloud.heusinger.art/s/sTAfyTdH3GbxQ2P/download/woko_dataset.zip",
    ),
]


def download_file(filename, url):
    r = requests.get(url, stream=True)

    if os.path.exists("./data"):
        print("data folder exists")
        os.chdir("./data")

    print(f"downloading {filename}")

    with open(filename, "wb") as f:
        total_length = int(r.headers.get("content-length"))
        for chunk in progress.bar(
            r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1
        ):
            if chunk:
                f.write(chunk)
                f.flush()


def yes_no_prompt(prompt_message):
    user_input = prompt.query(f"{prompt_message} (yes/no): ").lower()
    if user_input in ("yes", "y"):
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset Downloader",
        description="Downloads Datasets from https://rpg.ifi.uzh.ch/teaching.html",
        epilog="",
    )
    parser.add_argument("--all", help="download all datasets", action="store_true")
    args = parser.parse_args()

    selections = []
    if args.all:
        selections = [True] * len(targets)
    else:
        for i in range(len(targets)):
            selected = yes_no_prompt(f"Download dataset: {targets[i][0]}")
            selections.append(selected)

    for i in range(len(targets)):
        if selections[i]:
            download_file(targets[i][0], targets[i][1])
