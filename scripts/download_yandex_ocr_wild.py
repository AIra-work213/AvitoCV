"""Download Yandex's Russian OCR-in-the-wild benchmark from its original commit."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "yandex-cloud/ocr"
COMMIT = "6abc7e2c74aa4c9175236e3de14f15fd9785064b"
PREFIX = "ocr_comparison/rus_ocr_in_the_wild_dataset/"
TREE_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/train/rus_ocr_in_the_wild_dataset"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "avito-cv"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tree = fetch_json(TREE_URL)
    files = [
        item for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(PREFIX)
    ]
    if len(files) != 594:
        raise RuntimeError(f"Expected 594 files, found {len(files)}")

    def download(item: dict) -> tuple[str, int, bool]:
        relative = item["path"][len(PREFIX):]
        destination = args.output / relative
        if destination.is_file():
            data = destination.read_bytes()
            if git_blob_sha(data) == item["sha"]:
                return relative, len(data), True

        url = RAW_ROOT + urllib.parse.quote(item["path"])
        last_error: Exception | None = None
        for attempt in range(args.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "avito-cv"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    data = response.read()
                if git_blob_sha(data) != item["sha"]:
                    raise RuntimeError(f"Git SHA mismatch for {relative}")
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(destination)
                return relative, len(data), False
            except Exception as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to download {relative}") from last_error

    total_bytes = 0
    reused = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (relative, size, was_reused) in enumerate(
            pool.map(download, files), start=1
        ):
            total_bytes += size
            reused += int(was_reused)
            if index % 20 == 0 or index == len(files):
                print(
                    f"Verified {index}/{len(files)} files; "
                    f"{total_bytes / 1e6:.1f} MB; reused {reused}",
                    flush=True,
                )

    manifest = {
        "repository": f"https://github.com/{REPOSITORY}",
        "commit": COMMIT,
        "source_directory": PREFIX.rstrip("/"),
        "files": [
            {
                "path": item["path"][len(PREFIX):],
                "size": item["size"],
                "git_blob_sha1": item["sha"],
            }
            for item in files
        ],
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Complete: {len(files)} source files, {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
