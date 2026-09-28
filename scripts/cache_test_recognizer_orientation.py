#!/usr/bin/env python3
"""Сохраняет результаты OCR для исходных и повёрнутых тестовых изображений."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from paddlex import create_model
from tqdm import tqdm

FIELDS = [
    "image_id",
    "rec_score_direct",
    "rec_score_rotated",
    "rec_score_difference",
    "rec_text_direct",
    "rec_text_rotated",
]


def main() -> None:
    """Читает тест, запускает OCR пакетами и дописывает ещё не обработанные строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--images", type=Path, default=Path("data/test/test/images"))
    parser.add_argument(
        "--sample", type=Path, default=Path("data/test/sample_submission.csv")
    )
    args = parser.parse_args()
    with args.sample.open(newline="", encoding="utf-8") as file:
        image_ids = [row["image_id"] for row in csv.DictReader(file)]
    completed = set()
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            completed = {row["image_id"] for row in csv.DictReader(file)}
    model = create_model(args.model_name, device="cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if args.output.stat().st_size == 0:
            writer.writeheader()
        pending = [image_id for image_id in image_ids if image_id not in completed]
        for start in tqdm(
            range(0, len(pending), args.batch_size), desc=args.model_name
        ):
            batch_ids = pending[start : start + args.batch_size]
            images = [
                cv2.imread(str(args.images / f"{image_id}.png"))
                for image_id in batch_ids
            ]
            if any((image is None for image in images)):
                raise RuntimeError("Failed to read a test image")
            rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
            direct = [
                result.json["res"]
                for result in model.predict(images, batch_size=args.batch_size)
            ]
            flipped = [
                result.json["res"]
                for result in model.predict(rotated, batch_size=args.batch_size)
            ]
            for image_id, first, second in zip(batch_ids, direct, flipped, strict=True):
                score_direct = float(first["rec_score"])
                score_rotated = float(second["rec_score"])
                writer.writerow(
                    {
                        "image_id": image_id,
                        "rec_score_direct": score_direct,
                        "rec_score_rotated": score_rotated,
                        "rec_score_difference": score_rotated - score_direct,
                        "rec_text_direct": first["rec_text"],
                        "rec_text_rotated": second["rec_text"],
                    }
                )
            file.flush()
    print(f"Saved {len(image_ids)} rows to {args.output}")


if __name__ == "__main__":
    main()
