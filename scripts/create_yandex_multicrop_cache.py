#!/usr/bin/env python3
"""Сохраняет multi-crop признаки моделей на валидационной выборке Яндекса."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from paddlex import create_model
from rapidocr_onnxruntime import RapidOCR

POSITIONS = ("left", "center", "right")
REQUIRED_MODEL_FILES = ("inference.yml", "inference.json", "inference.pdiparams")


def parse_args() -> argparse.Namespace:
    """Читает пути и параметры построения multi-crop кеша."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train/yandex_ocr_real_long_lines_v4/labels.csv"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/yandex_real_long_v4_multicrop_raw.csv"),
    )
    parser.add_argument("--threshold-ratio", type=float, default=10.0)
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
    return parser.parse_args()


def crop_at(image: np.ndarray, target_ratio: float, position: str) -> np.ndarray:
    """Вырезает окно заданной ширины слева, по центру или справа."""
    height, width = image.shape[:2]
    crop_width = min(width, max(1, round(height * target_ratio)))
    if position == "left":
        left = 0
    elif position == "right":
        left = width - crop_width
    elif position == "center":
        left = (width - crop_width) // 2
    else:
        raise ValueError(position)
    return image[:, left : left + crop_width]


def paddle_probability(result: object) -> float:
    """Преобразует ответ PaddleX в вероятность поворота на 180 градусов."""
    payload = result.json["res"]
    score = float(payload["scores"][0])
    return score if payload["label_names"][0] == "180_degree" else 1.0 - score


def paddle_tta(model: object, images: list[np.ndarray], batch_size: int) -> np.ndarray:
    """Усредняет предсказания PaddleX для двух поворотов изображения."""
    direct = np.asarray(
        [
            paddle_probability(result)
            for result in model.predict(images, batch_size=batch_size)
        ]
    )
    rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
    flipped = np.asarray(
        [
            paddle_probability(result)
            for result in model.predict(rotated, batch_size=batch_size)
        ]
    )
    return 0.5 * (direct + 1.0 - flipped)


def rapid_probabilities(engine: RapidOCR, images: list[np.ndarray]) -> np.ndarray:
    """Получает вероятности ориентации от RapidOCR."""
    result = engine.text_cls(images)
    if isinstance(result, tuple):
        result = result[1] if len(result) == 3 else result[0]
    return np.asarray(
        [
            float(score) if label == "180" else 1.0 - float(score)
            for label, score in result
        ]
    )


def load_orientation_model(model_name: str, model_dir: Path) -> object:
    """Загружает локальную модель ориентации или модель из кеша PaddleX."""
    if model_dir.is_dir() and all(
        ((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES)
    ):
        print(f"Используется локальная модель: {model_dir}", flush=True)
        return create_model(model_name, model_dir=str(model_dir), device="cpu")
    print(f"PaddleX автоматически скачает модель {model_name}", flush=True)
    return create_model(model_name, device="cpu")


def main() -> None:
    """Считает признаки трёх моделей для окон широких строк."""
    args = parse_args()
    labels = pd.read_csv(args.labels)
    labels["ratio"] = labels["output_ratio"]
    baseline_frame = pd.read_csv(args.baseline).set_index("image_id")
    baseline = np.asarray(
        [baseline_frame.loc[image_id, "p_final"] for image_id in labels.image_id],
        dtype=float,
    )
    wide = np.flatnonzero(labels.ratio.to_numpy(float) >= args.threshold_ratio)
    image_paths = [Path(path) for path in labels.image_path]
    images = [cv2.imread(str(image_paths[index])) for index in wide]
    unreadable = [
        str(image_paths[index]) for index, image in zip(wide, images) if image is None
    ]
    if unreadable:
        raise RuntimeError(f"Не удалось прочитать изображения: {unreadable[:5]}")
    x025 = load_orientation_model("PP-LCNet_x0_25_textline_ori", args.x025_model_dir)
    x1 = load_orientation_model("PP-LCNet_x1_0_textline_ori", args.x1_model_dir)
    rapid = RapidOCR()
    rows: dict[str, object] = {
        "image_id": labels.image_id.iloc[wide].to_numpy(),
        "target": labels.target.iloc[wide].to_numpy(int),
        "ratio": labels.ratio.iloc[wide].to_numpy(float),
        "p_baseline": baseline[wide],
    }
    for position in POSITIONS:
        paddle_crops = [crop_at(image, 2.0, position) for image in images]
        rapid_crops = [crop_at(image, 4.0, position) for image in images]
        rows[f"x025_{position}"] = paddle_tta(x025, paddle_crops, args.batch_size)
        rows[f"x1_{position}"] = paddle_tta(x1, paddle_crops, args.batch_size)
        rows[f"rapid_{position}"] = rapid_probabilities(rapid, rapid_crops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Сохранено {len(wide)} строк: {args.output}")


if __name__ == "__main__":
    main()
