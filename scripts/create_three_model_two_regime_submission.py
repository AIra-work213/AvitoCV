#!/usr/bin/env python3
"""Rebuild the exact three-model, two-regime orientation submission.

The output is the base orientation feature consumed by
``create_submission_ocr_v5_meta.py``.  All intermediate model predictions are
recomputed from the test images; no historical submission files are required.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(".paddlex-cache").resolve()))

import cv2
import numpy as np
import pandas as pd
from paddlex import create_model
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm


POSITIONS = ("left", "center", "right")
VALID_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Создать базовый two-regime сабмит из исходных изображений"
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/test/sample_submission.csv"),
    )
    parser.add_argument("--images", type=Path, default=Path("data/test/test/images"))
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
        "--output",
        type=Path,
        default=Path("submission_three_model_two_regime_tuned.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--x025-cache",
        type=Path,
        help="Необязательный ранее рассчитанный CSV image_id,p_180",
    )
    parser.add_argument(
        "--rapid-cache",
        type=Path,
        help="Необязательный ранее рассчитанный orientation_results.json",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "p_180"]:
            raise ValueError(f"Неверные колонки в {path}: {reader.fieldnames}")
        ids = [row["image_id"] for row in reader]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Повторяющиеся image_id в {path}")
    return ids


def index_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f"Повторяющееся имя изображения: {path.stem}")
        result[path.stem] = path
    return result


def create_orientation_model(name: str, directory: Path) -> object:
    required = ("inference.yml", "inference.json", "inference.pdiparams")
    if directory.is_dir() and all((directory / filename).is_file() for filename in required):
        print(f"Используется локальная модель: {directory}", flush=True)
        return create_model(name, model_dir=str(directory), device="cpu")
    print(f"PaddleX загрузит отсутствующую модель {name}", flush=True)
    return create_model(name, device="cpu")


def paddle_probability(result: object) -> float:
    payload = result.json["res"]
    score = float(payload["scores"][0])
    return score if payload["label_names"][0] == "180_degree" else 1.0 - score


def paddle_tta(model: object, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    direct = np.asarray(
        [paddle_probability(result) for result in model.predict(images, batch_size=batch_size)],
        dtype=np.float64,
    )
    rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
    flipped = np.asarray(
        [paddle_probability(result) for result in model.predict(rotated, batch_size=batch_size)],
        dtype=np.float64,
    )
    return 0.5 * (direct + 1.0 - flipped)


def paddle_path_tta(
    model: object, paths: list[Path], batch_size: int, description: str
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=description):
        batch = paths[start : start + batch_size]
        direct = np.asarray(
            [
                paddle_probability(result)
                for result in model.predict([str(path) for path in batch], batch_size=batch_size)
            ],
            dtype=np.float64,
        )
        images = [cv2.imread(str(path)) for path in batch]
        if any(image is None for image in images):
            raise RuntimeError("Не удалось прочитать одно из тестовых изображений")
        rotated = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
        flipped = np.asarray(
            [paddle_probability(result) for result in model.predict(rotated, batch_size=batch_size)],
            dtype=np.float64,
        )
        output.append(0.5 * (direct + 1.0 - flipped))
    return np.concatenate(output)


def parse_rapid(result: object) -> object:
    if isinstance(result, tuple):
        if len(result) == 3:
            return result[1]
        if len(result) == 2:
            return result[0]
        return result[0]
    return result


def rapid_probabilities(engine: RapidOCR, images: list[np.ndarray]) -> np.ndarray:
    result = parse_rapid(engine.text_cls(images))
    return np.asarray(
        [float(score) if label == "180" else 1.0 - float(score) for label, score in result],
        dtype=np.float64,
    )


def temperature(probabilities: np.ndarray, value: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(logits / value, -700.0, 700.0)))


def inverse_temperature(probabilities: np.ndarray, value: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = value * np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def crop_at(image: np.ndarray, target_ratio: float, position: str) -> np.ndarray:
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


def load_x025_cache(path: Path, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["image_id", "p_180"] or frame.image_id.tolist() != ids:
        raise ValueError(f"Кеш {path} не соответствует sample_submission.csv")
    return frame.p_180.to_numpy(dtype=np.float64)


def load_rapid_cache(path: Path, ids: list[str]) -> np.ndarray:
    import json

    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup = {Path(row["filename"]).stem: float(row["prob_180"]) for row in rows}
    if set(lookup) != set(ids):
        raise ValueError(f"Кеш {path} не соответствует sample_submission.csv")
    return np.asarray([lookup[image_id] for image_id in ids], dtype=np.float64)


def full_image_predictions(
    ids: list[str],
    paths: list[Path],
    x025_model: object,
    x1_model: object,
    rapid_model: RapidOCR,
    batch_size: int,
    x025_cache: Path | None,
    rapid_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x025_cache:
        x025 = load_x025_cache(x025_cache, ids)
    else:
        x025_raw = paddle_path_tta(x025_model, paths, batch_size, "PP-LCNet x0.25")
        # The historical x0.25 cache was written with ten decimal places.
        x025 = np.asarray([float(f"{value:.10f}") for value in x025_raw])

    x1 = paddle_path_tta(x1_model, paths, batch_size, "PP-LCNet x1.0")

    if rapid_cache:
        rapid = load_rapid_cache(rapid_cache, ids)
    else:
        values: list[float] = []
        for start in tqdm(range(0, len(paths), 32), desc="RapidOCR"):
            images = [cv2.imread(str(path)) for path in paths[start : start + 32]]
            if any(image is None for image in images):
                raise RuntimeError("Не удалось прочитать одно из тестовых изображений")
            values.extend(rapid_probabilities(rapid_model, images).tolist())
        # orientation_results.json stored six-decimal probabilities.
        rapid = np.asarray([round(value, 6) for value in values], dtype=np.float64)
    return x025, x1, rapid


def historical_regular_probability(
    x025: np.ndarray, x1: np.ndarray, rapid: np.ndarray
) -> np.ndarray:
    # The selected regular regime was reconstructed from the ten-decimal CSV of
    # the preceding ensemble.  Repeating that quantization preserves its bytes.
    old = 0.125 * x025 + 0.750 * temperature(x1, 0.05) + 0.125 * rapid
    old = np.asarray([float(f"{value:.10f}") for value in old])
    x1_t005 = (old - 0.125 * x025 - 0.125 * rapid) / 0.750
    recovered_x1 = inverse_temperature(np.clip(x1_t005, 0.0, 1.0), 0.05)
    return temperature(
        0.20 * x025 + 0.55 * temperature(recovered_x1, 0.02) + 0.25 * rapid,
        0.5,
    )


def multicrop_probability(
    items: list[tuple[int, np.ndarray]],
    x025_model: object,
    x1_model: object,
    rapid_model: RapidOCR,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([index for index, _ in items], dtype=int)
    windows: list[np.ndarray] = []
    for position in POSITIONS:
        values: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(items), batch_size), desc=f"{label}: {position}"
        ):
            batch = items[start : start + batch_size]
            pp = [crop_at(image, 2.0, position) for _, image in batch]
            rr = [crop_at(image, 4.0, position) for _, image in batch]
            p025 = paddle_tta(x025_model, pp, batch_size=32)
            p1 = paddle_tta(x1_model, pp, batch_size=32)
            rapid = rapid_probabilities(rapid_model, rr)
            values.append(
                0.05 * p025 + 0.65 * temperature(p1, 0.02) + 0.30 * rapid
            )
        windows.append(np.concatenate(values))
    probability = temperature(np.mean(np.stack(windows, axis=1), axis=1), 0.5)
    return indices, probability


def main() -> None:
    args = parse_args()
    if args.batch_size != 32:
        raise ValueError("Для побайтового воспроизведения требуется --batch-size 32")

    ids = read_ids(args.sample_submission)
    image_by_id = index_images(args.images)
    if set(image_by_id) != set(ids):
        missing = len(set(ids) - set(image_by_id))
        extra = len(set(image_by_id) - set(ids))
        raise ValueError(f"Изображения не совпадают с шаблоном: missing={missing}, extra={extra}")
    paths = [image_by_id[image_id] for image_id in ids]

    x025_model = create_orientation_model(
        "PP-LCNet_x0_25_textline_ori", args.x025_model_dir
    )
    x1_model = create_orientation_model("PP-LCNet_x1_0_textline_ori", args.x1_model_dir)
    rapid_model = RapidOCR()

    x025, x1, rapid = full_image_predictions(
        ids,
        paths,
        x025_model,
        x1_model,
        rapid_model,
        args.batch_size,
        args.x025_cache,
        args.rapid_cache,
    )
    final = historical_regular_probability(x025, x1, rapid)

    ratio_gt_10: list[tuple[int, np.ndarray]] = []
    ratio_eq_10: list[tuple[int, np.ndarray]] = []
    for index, path in enumerate(tqdm(paths, desc="Чтение размеров")):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Не удалось прочитать {path}")
        ratio = image.shape[1] / image.shape[0]
        if ratio > 10.0:
            ratio_gt_10.append((index, image))
        elif ratio == 10.0:
            ratio_eq_10.append((index, image))

    # These subsets were historically inferred in separate runs.
    for label, items in (("ratio > 10", ratio_gt_10), ("ratio = 10", ratio_eq_10)):
        indices, probability = multicrop_probability(
            items, x025_model, x1_model, rapid_model, args.batch_size, label
        )
        final[indices] = probability

    if len(ids) != 20_000 or len(ratio_gt_10) != 2_806 or len(ratio_eq_10) != 16:
        raise RuntimeError(
            "Неожиданный состав теста: "
            f"rows={len(ids)}, ratio>10={len(ratio_gt_10)}, ratio=10={len(ratio_eq_10)}"
        )
    if not np.all(np.isfinite(final)) or np.any((final < 0.0) | (final > 1.0)):
        raise RuntimeError("Получены некорректные вероятности")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": ids, "p_180": final}).to_csv(
        args.output, index=False, float_format="%.10f"
    )
    print(f"Готово: {args.output} ({len(ids)} строк)")


if __name__ == "__main__":
    main()
