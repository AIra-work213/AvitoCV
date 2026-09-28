#!/usr/bin/env python3
"""Оценивает ансамбль двух PP-LCNet и классификатора RapidOCR."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(".paddlex-cache").resolve()))
import cv2
import numpy as np
from paddlex import create_model
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Читает пути к данным, моделям и выходному файлу."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train/yandex_ocr_orientation_eval/labels.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--x025-model-dir",
        type=Path,
        default=Path("models/PP-LCNet_x0_25_textline_ori_infer"),
    )
    parser.add_argument(
        "--x1-model-dir",
        type=Path,
        default=Path("models/PP-LCNet_x1_0_textline_ori_infer"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/yandex_ocr_three_model_eval.csv")
    )
    return parser.parse_args()


def create_orientation_model(model_name: str, model_dir: Path) -> object:
    """Загружает локальную модель или скачивает официальную модель PaddleX."""
    required = ("inference.yml", "inference.json", "inference.pdiparams")
    if model_dir.is_dir() and all(((model_dir / name).is_file() for name in required)):
        print(f"Using local model: {model_dir}", flush=True)
        return create_model(model_name, model_dir=str(model_dir), device="cpu")
    print(
        f"Local model is missing or incomplete: {model_dir}. "
        f"PaddleX will download {model_name} automatically.",
        flush=True,
    )
    return create_model(model_name, device="cpu")


def paddle_probability(result: object) -> float:
    """Преобразует ответ PaddleX в вероятность поворота на 180 градусов."""
    payload = result.json["res"]
    score = float(payload["scores"][0])
    return score if payload["label_names"][0] == "180_degree" else 1.0 - score


def rapid_probabilities(engine: RapidOCR, images: list[np.ndarray]) -> np.ndarray:
    """Преобразует ответы RapidOCR в вероятности поворота."""
    result = engine.text_cls(images)
    if isinstance(result, tuple):
        if len(result) == 3:
            result = result[1]
        elif len(result) == 2:
            result = result[0]
    values = []
    for label, score in result:
        score = float(score)
        values.append(score if label == "180" else 1.0 - score)
    return np.asarray(values, dtype=np.float64)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Применяет температурную калибровку в пространстве логитов."""
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -700, 700)))


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Считает Brier score, 1 − Brier и accuracy."""
    brier = float(np.mean(np.square(y_prob - y_true)))
    return {
        "one_minus_brier": 1.0 - brier,
        "brier_score": brier,
        "accuracy": float(np.mean((y_prob >= 0.5) == y_true)),
        "errors": int(np.sum((y_prob >= 0.5) != y_true)),
    }


def main() -> None:
    """Запускает три модели, объединяет вероятности и сохраняет отчёт."""
    args = parse_args()
    with args.labels.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    x025_model = create_orientation_model(
        "PP-LCNet_x0_25_textline_ori", args.x025_model_dir
    )
    x1_model = create_orientation_model("PP-LCNet_x1_0_textline_ori", args.x1_model_dir)
    rapid_model = RapidOCR()
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Evaluation"):
        batch = rows[start : start + args.batch_size]
        images = [cv2.imread(row["image_path"]) for row in batch]
        if any((image is None for image in images)):
            raise RuntimeError("Failed to read a generated crop")
        rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]

        def paddle_tta(model: object) -> np.ndarray:
            """Усредняет предсказания PaddleX для исходного и повёрнутого пакета."""
            direct = np.asarray(
                [
                    paddle_probability(result)
                    for result in model.predict(images, batch_size=args.batch_size)
                ]
            )
            flipped = np.asarray(
                [
                    paddle_probability(result)
                    for result in model.predict(rotated, batch_size=args.batch_size)
                ]
            )
            return 0.5 * (direct + 1.0 - flipped)

        p_x025 = paddle_tta(x025_model)
        p_x1 = paddle_tta(x1_model)
        p_x1_calibrated = temperature_scale(p_x1, 0.05)
        p_rapid = rapid_probabilities(rapid_model, images)
        p_final = 0.125 * p_x025 + 0.75 * p_x1_calibrated + 0.125 * p_rapid
        for row, p0, p1, p1c, pr, pf in zip(
            batch, p_x025, p_x1, p_x1_calibrated, p_rapid, p_final, strict=True
        ):
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "target": int(row["target"]),
                    "p_x025_tta": float(p0),
                    "p_x1_tta": float(p1),
                    "p_x1_tta_t005": float(p1c),
                    "p_rapid": float(pr),
                    "p_final": float(pf),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    y_true = np.asarray([row["target"] for row in predictions], dtype=np.float64)
    report = {
        "samples": len(predictions),
        "x025_flip_tta": metrics(
            y_true, np.asarray([row["p_x025_tta"] for row in predictions])
        ),
        "x1_flip_tta": metrics(
            y_true, np.asarray([row["p_x1_tta"] for row in predictions])
        ),
        "x1_flip_tta_t005": metrics(
            y_true, np.asarray([row["p_x1_tta_t005"] for row in predictions])
        ),
        "rapid": metrics(y_true, np.asarray([row["p_rapid"] for row in predictions])),
        "ensemble": metrics(
            y_true, np.asarray([row["p_final"] for row in predictions])
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
