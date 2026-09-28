#!/usr/bin/env python3
"""Собирает финальный сабмит метамодели OCR-v5 из сохранённых предсказаний."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260927
C_VALUE = 0.3
TEMPERATURE = 0.42
FEATURE_COLUMNS = [*range(12), 18, 20, 21]
ALPHABET = " абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz0123456789-"
OCR_COLUMNS = [
    "image_id",
    "rec_score_direct",
    "rec_score_rotated",
    "rec_score_difference",
    "rec_text_direct",
    "rec_text_rotated",
]
YANDEX_LABELS = Path("data/train/yandex_ocr_real_long_lines_v4/labels.csv")
YANDEX_BASE = Path("outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv")
YANDEX_MULTICROP = Path("outputs/yandex_real_long_v4_multicrop_raw.csv")
YANDEX_CYRILLIC = Path("outputs/yandex_cyrillic_v5_rec_predictions.csv")
YANDEX_ESLAV = Path("outputs/yandex_eslav_v5_rec_predictions.csv")
YANDEX_ENGLISH = Path("outputs/yandex_english_v5_rec_predictions.csv")


def normalize(text: object) -> str:
    """Приводит распознанный текст к алфавиту символьной модели."""
    normalized = str(text).lower().replace("ё", "е")
    normalized = "".join(
        character if character in ALPHABET else " " for character in normalized
    )
    return re.sub(r" +", " ", normalized).strip()


class CharLM:
    """Оценивает текст сглаженной символьной триграммной моделью."""

    def __init__(self, texts: list[str], alpha: float = 0.2) -> None:
        """Подсчитывает символьные контексты и триграммы в корпусе."""
        self.alpha = alpha
        self.context = Counter()
        self.ngram = Counter()
        self.vocabulary = len(ALPHABET) + 2

        for text in texts:
            value = "^^" + normalize(text) + "$$"
            for index in range(2, len(value)):
                self.context[value[index - 2 : index]] += 1
                self.ngram[value[index - 2 : index + 1]] += 1

    def score(self, text: object) -> float:
        """Возвращает среднюю сглаженную логарифмическую вероятность текста."""
        value = "^^" + normalize(text) + "$$"
        if len(value) <= 4:
            return -12.0

        scores = [
            math.log(
                (self.ngram[value[index - 2 : index + 1]] + self.alpha)
                / (
                    self.context[value[index - 2 : index]]
                    + self.alpha * self.vocabulary
                )
            )
            for index in range(2, len(value))
        ]
        return float(np.mean(scores))


def corpus() -> list[str]:
    """Читает пригодные транскрипции из скачанного OCR-датасета Яндекса."""
    texts: list[str] = []
    source = Path("data/train/rus_ocr_in_the_wild_dataset")
    for path in source.glob("gt_img_*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split(",", 8)
            if len(parts) == 9 and parts[8].strip() != "###":
                texts.append(parts[8].strip())
    return texts


def apply_temp(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Применяет температурную калибровку вероятностей в пространстве логитов."""
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return 1 / (1 + np.exp(-np.log(probabilities / (1 - probabilities)) / temperature))


def attach(frame: pd.DataFrame, path: Path, prefix: str) -> pd.DataFrame:
    """Добавляет к таблице кеш одной OCR-модели и переименовывает её признаки."""
    extra = pd.read_csv(path)[OCR_COLUMNS].rename(
        columns={
            column: f"{prefix}_{column}"
            for column in OCR_COLUMNS
            if column != "image_id"
        }
    )
    return frame.merge(extra, on="image_id")


def add_ocr(
    frame: pd.DataFrame,
    cyrillic_path: Path,
    eslav_path: Path,
    english_path: Path,
) -> pd.DataFrame:
    """Добавляет три OCR-кеша в исходном порядке объединения."""
    frame = attach(frame, cyrillic_path, "cyr")
    frame = attach(frame, eslav_path, "eslav")
    return attach(frame, english_path, "eng")


def one_model(row: Any, prefix: str, language_model: CharLM) -> list[float]:
    """Строит шесть признаков ориентации по одной OCR-модели."""
    direct = normalize(getattr(row, f"{prefix}_rec_text_direct"))
    rotated = normalize(getattr(row, f"{prefix}_rec_text_rotated"))
    cyrillic_direct = sum("а" <= char <= "я" for char in direct) / max(1, len(direct))
    cyrillic_rotated = sum("а" <= char <= "я" for char in rotated) / max(
        1, len(rotated)
    )
    return [
        getattr(row, f"{prefix}_rec_score_difference"),
        getattr(row, f"{prefix}_rec_score_direct"),
        getattr(row, f"{prefix}_rec_score_rotated"),
        np.tanh((len(rotated) - len(direct)) / 5),
        language_model.score(rotated) - language_model.score(direct),
        cyrillic_rotated - cyrillic_direct,
    ]


def features(frame: pd.DataFrame, language_model: CharLM) -> np.ndarray:
    """Строит полную матрицу признаков в исходном порядке столбцов."""
    output: list[list[float]] = []
    for row in frame.itertuples():
        cyrillic = one_model(row, "cyr", language_model)
        eslav = one_model(row, "eslav", language_model)
        english = one_model(row, "eng", language_model)
        current_probability = np.clip(row.p_current, 1e-6, 1 - 1e-6)
        score_differences = [cyrillic[0], eslav[0], english[0]]
        output.append(
            cyrillic
            + eslav
            + english
            + [
                np.log(current_probability / (1 - current_probability)),
                np.log(max(row.ratio, 1e-3)),
                max(score_differences) - min(score_differences),
                np.mean(score_differences),
            ]
        )
    return np.asarray(output, float)


def yandex_frame() -> pd.DataFrame:
    """Собирает обучающую таблицу метамодели из зафиксированных кешей Яндекса."""
    labels = pd.read_csv(YANDEX_LABELS).set_index("image_id")
    base = pd.read_csv(YANDEX_BASE).set_index("image_id")
    multicrop = pd.read_csv(YANDEX_MULTICROP).set_index("image_id")

    probability = apply_temp(
        0.20 * base.p_x025_tta.to_numpy()
        + 0.55 * apply_temp(base.p_x1_tta.to_numpy(), 0.02)
        + 0.25 * base.p_rapid.to_numpy(),
        0.5,
    )
    wide_mask = labels.output_ratio.to_numpy(float) >= 10

    for location, image_id in zip(np.flatnonzero(wide_mask), labels.index[wide_mask]):
        row = multicrop.loc[image_id]
        windows = [
            0.05 * row[f"x025_{position}"]
            + 0.65 * apply_temp(np.asarray([row[f"x1_{position}"]]), 0.02)[0]
            + 0.30 * row[f"rapid_{position}"]
            for position in ("left", "center", "right")
        ]
        probability[location] = apply_temp(np.asarray([np.mean(windows)]), 0.5)[0]

    return pd.DataFrame(
        {
            "image_id": labels.index,
            "target": labels.target,
            "ratio": labels.output_ratio,
            "p_current": probability,
        }
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Читает пути и параметры для сборки финального сабмита."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/test/sample_submission.csv"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/test/test/images"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("submission_three_model_two_regime_tuned.csv"),
    )
    parser.add_argument(
        "--cyr",
        type=Path,
        default=Path("outputs/test_cyrillic_v5_rec_predictions_b8.csv"),
    )
    parser.add_argument(
        "--eslav",
        type=Path,
        default=Path("outputs/test_eslav_v5_rec_predictions_b8.csv"),
    )
    parser.add_argument(
        "--eng",
        type=Path,
        default=Path("outputs/test_english_v5_rec_predictions_b8.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_ocr_v5_meta_yandex_selected_b8.csv"),
    )
    return parser.parse_args()


def test_frame(args: argparse.Namespace, image_ids: list[str]) -> pd.DataFrame:
    """Собирает метаданные теста, сохраняя порядок строк sample submission."""
    base = pd.read_csv(args.base)
    if base.image_id.tolist() != image_ids:
        raise ValueError("Base predictions do not match sample order")

    ratios: list[float] = []
    for image_id in image_ids:
        image = cv2.imread(str(args.images / f"{image_id}.png"))
        if image is None:
            raise RuntimeError(f"Cannot read {image_id}")
        ratios.append(image.shape[1] / image.shape[0])

    return pd.DataFrame(
        {
            "image_id": image_ids,
            "ratio": ratios,
            "p_current": base.p_180,
        }
    )


def main() -> None:
    """Обучает зафиксированную метамодель и записывает сабмит с отчётом."""
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    image_ids = pd.read_csv(args.sample).image_id.tolist()
    if len(image_ids) != 20000 or len(set(image_ids)) != 20000:
        raise ValueError("Expected 20,000 unique IDs")

    language_model = CharLM(corpus())
    train = add_ocr(
        yandex_frame(),
        YANDEX_CYRILLIC,
        YANDEX_ESLAV,
        YANDEX_ENGLISH,
    )
    test = add_ocr(
        test_frame(args, image_ids),
        args.cyr,
        args.eslav,
        args.eng,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_VALUE,
            max_iter=4000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    model.fit(
        features(train, language_model)[:, FEATURE_COLUMNS],
        train.target.to_numpy(int),
    )
    probability = apply_temp(
        model.predict_proba(features(test, language_model)[:, FEATURE_COLUMNS])[:, 1],
        TEMPERATURE,
    )

    result = pd.DataFrame({"image_id": image_ids, "p_180": probability})
    result.to_csv(args.output, index=False, float_format="%.10f")
    report = {
        "seed": SEED,
        "model_selection": "Yandex source-group OOF",
        "meta_training_rows": len(train),
        "feature_variant": "russian_only",
        "C": C_VALUE,
        "temperature": TEMPERATURE,
        "ocr_batch_size": 8,
        "submission": {
            "path": str(args.output),
            "rows": len(result),
            "min": float(probability.min()),
            "max": float(probability.max()),
            "mean": float(probability.mean()),
        },
    }
    report_text = json.dumps(report, indent=2)
    args.output.with_suffix(".json").write_text(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    main()
