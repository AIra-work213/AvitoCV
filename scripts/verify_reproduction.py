#!/usr/bin/env python3
"""Проверяет все значимые артефакты полного DVC-пайплайна."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd

EXPECTED_IMAGE_SET_SHA256 = (
    "dc1cdaa3b4e97cdf5bf396faafb9c115b2676642e603ef29c52692e0e62bc636"
)
EXPECTED_SHA256 = {
    "data/train/rus_ocr_in_the_wild_dataset/source_manifest.json": (
        "bd3b12f1095a2cff7e4227350ba250dc65374ea846d050aeab94b6125c3939a7"
    ),
    "data/train/yandex_ocr_real_long_lines_v4/labels.csv": (
        "aeadb9465297e359afe436c77091b47b2184b82ecf4912a8060f554bc9476fb6"
    ),
    "data/train/yandex_ocr_real_long_lines_v4/summary.json": (
        "2d3cbd689cb2c3cc23f64016f3ec9a5f1ccf1f2f84014a8ec2d6578af42ef48b"
    ),
    "data/train/yandex_ocr_real_long_lines_v4/contact_sheet.jpg": (
        "ea57e5eafcdd936a31d6d1cb2359f5b749c305716f4b70e7964fd0ddba009ca1"
    ),
    "outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv": (
        "76f629d5880146fcf543d49f3937b29b23943b4937b5a23ddce7dac574a968dc"
    ),
    "outputs/yandex_real_long_v4_multicrop_raw.csv": (
        "f54593539769d48cfa3bfcf2bff5e7a5583e71940d9f19daf6ac234fe2f8a997"
    ),
    "outputs/yandex_cyrillic_v5_rec_predictions.csv": (
        "95e44cd20877e8f22aac3338bb325b0642d1bd3dd84091ccc4f461fc1e6c59aa"
    ),
    "outputs/yandex_eslav_v5_rec_predictions.csv": (
        "6970aeb597757db28f360bcba1b7a71c845b4e840e11de46b89b1300a814d968"
    ),
    "outputs/yandex_english_v5_rec_predictions.csv": (
        "f90627b9d12474736353ec37cca0a6b13ee3fb611e5849504ed004f51b3759a4"
    ),
    "submission_three_model_two_regime_tuned.csv": (
        "fdcb10ab338a74fd28df7e6809e7e1f21aa85cd23a449df0449a40e11e69dfba"
    ),
    "outputs/test_cyrillic_v5_rec_predictions_b8.csv": (
        "4de3f63e77eccd5fe5abeb8f5f6469986e654fe30fe4b42f9098fc96408905b1"
    ),
    "outputs/test_eslav_v5_rec_predictions_b8.csv": (
        "2390432d09f9e747114a7a93ed1d17893bfd6b8310f8c3a9762092eb6601925b"
    ),
    "outputs/test_english_v5_rec_predictions_b8.csv": (
        "242d0063a8e4b72b7f8d9e612bd7d0fa8505bbd42b48f11be5fc5418fc5d358c"
    ),
    "submission_ocr_v5_meta_yandex_selected_b8.csv": (
        "9580d7080b3e19c0b3e01e80e2c291743739caa2964a6dd049bbb9cc99226ffd"
    ),
}


def parse_args() -> argparse.Namespace:
    """Читает путь к итоговому отчёту проверки."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reproduction_check.json"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Возвращает SHA-256 содержимого одного файла."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def image_set_sha256(directory: Path) -> str:
    """Хеширует имена и байты всех подготовленных PNG по порядку."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.png")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_submission(path: Path) -> None:
    """Проверяет формат, размер и диапазон вероятностей сабмита."""
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["image_id", "p_180"]:
        raise RuntimeError(f"Неверные столбцы сабмита: {frame.columns.tolist()}")
    if len(frame) != 20_000 or frame.image_id.nunique() != 20_000:
        raise RuntimeError("Сабмит должен содержать 20 000 уникальных image_id")
    if not frame.p_180.between(0.0, 1.0).all():
        raise RuntimeError("Вероятности сабмита вышли за диапазон [0, 1]")


def main() -> None:
    """Сверяет хеши и создаёт отчёт только после полного совпадения."""
    args = parse_args()
    actual = {}
    mismatches = []
    for name, expected in EXPECTED_SHA256.items():
        path = Path(name)
        if not path.is_file():
            mismatches.append(f"нет файла {name}")
            continue
        digest = file_sha256(path)
        actual[name] = digest
        if digest != expected:
            mismatches.append(f"{name}: получено {digest}, ожидалось {expected}")

    image_digest = image_set_sha256(
        Path("data/train/yandex_ocr_real_long_lines_v4/images")
    )
    if image_digest != EXPECTED_IMAGE_SET_SHA256:
        mismatches.append(
            "Yandex PNG: получено "
            f"{image_digest}, ожидалось {EXPECTED_IMAGE_SET_SHA256}"
        )

    validate_submission(Path("submission_ocr_v5_meta_yandex_selected_b8.csv"))
    if mismatches:
        raise RuntimeError("Побайтовая проверка не пройдена:\n" + "\n".join(mismatches))

    report = {
        "status": "ok",
        "opencv": cv2.__version__,
        "yandex_image_set_sha256": image_digest,
        "files": actual,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Все артефакты совпали с эталонными SHA-256.")


if __name__ == "__main__":
    main()
