#!/usr/bin/env python3
"""Оценивает ориентацию строк по уверенности OCR в двух поворотах."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from paddlex import create_model
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold


def probability(difference: np.ndarray, temperature: float) -> np.ndarray:
    """Преобразует разницу уверенностей OCR в вероятность через температуру."""
    return 1.0 / (1.0 + np.exp(-np.clip(difference / temperature, -40.0, 40.0)))


def fit_temperature(difference: np.ndarray, target: np.ndarray) -> float:
    """Подбирает температуру, минимизирующую Brier loss."""
    result = minimize_scalar(
        lambda log_t: np.mean((probability(difference, np.exp(log_t)) - target) ** 2),
        bounds=(np.log(0.01), np.log(2.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Считает Brier score, 1 − Brier и accuracy."""
    brier = float(np.mean((prediction - target) ** 2))
    errors = int(np.sum((prediction >= 0.5) != target))
    return {
        "one_minus_brier": 1.0 - brier,
        "brier_score": brier,
        "accuracy": 1.0 - errors / len(target),
        "errors": errors,
    }


def main() -> None:
    """Считает OCR-признаки, OOF-калибровку и итоговый отчёт."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="cyrillic_PP-OCRv3_mobile_rec")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/cyrillic_PP-OCRv3_mobile_rec_infer"),
    )
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    kwargs = {"device": "cpu"}
    if args.model_name == "cyrillic_PP-OCRv3_mobile_rec" and args.model_dir.exists():
        kwargs["model_dir"] = str(args.model_dir)
    model = create_model(args.model_name, **kwargs)
    direct_scores, rotated_scores = ([], [])
    direct_texts, rotated_texts = ([], [])
    started = time.perf_counter()
    for start in range(0, len(labels), args.batch_size):
        rows = labels.iloc[start : start + args.batch_size]
        images = [cv2.imread(path) for path in rows["image_path"]]
        if any((image is None for image in images)):
            raise RuntimeError("Failed to read an image")
        rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
        direct = [
            result.json["res"]
            for result in model.predict(images, batch_size=args.batch_size)
        ]
        flipped = [
            result.json["res"]
            for result in model.predict(rotated, batch_size=args.batch_size)
        ]
        direct_scores.extend((float(item["rec_score"]) for item in direct))
        rotated_scores.extend((float(item["rec_score"]) for item in flipped))
        direct_texts.extend((item["rec_text"] for item in direct))
        rotated_texts.extend((item["rec_text"] for item in flipped))
    direct_score = np.asarray(direct_scores)
    rotated_score = np.asarray(rotated_scores)
    difference = rotated_score - direct_score
    target = labels["target"].to_numpy(dtype=float)
    groups = labels["source_image"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(labels))
    fold_temperatures = []
    for train, validation in splitter.split(difference, target, groups):
        temperature = fit_temperature(difference[train], target[train])
        fold_temperatures.append(temperature)
        oof[validation] = probability(difference[validation], temperature)
    full_temperature = fit_temperature(difference, target)
    fitted = probability(difference, full_temperature)
    result = pd.DataFrame(
        {
            "image_id": labels["image_id"],
            "target": labels["target"],
            "rec_score_direct": direct_score,
            "rec_score_rotated": rotated_score,
            "rec_score_difference": difference,
            "p_rec_oof": oof,
            "p_rec_full_fit": fitted,
            "rec_text_direct": direct_texts,
            "rec_text_rotated": rotated_texts,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    report = {
        "samples": len(labels),
        "oof": metrics(target, oof),
        "fit_on_all": metrics(target, fitted),
        "full_temperature": full_temperature,
        "fold_temperatures": fold_temperatures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
