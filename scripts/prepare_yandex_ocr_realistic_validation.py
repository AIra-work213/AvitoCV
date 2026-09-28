#!/usr/bin/env python3
"""Добавляет контролируемые искажения в OCR-валидацию Яндекса."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Сжимает изображение в JPEG и сразу декодирует его обратно."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def resize_roundtrip(image: np.ndarray, scale: float) -> np.ndarray:
    """Уменьшает изображение и восстанавливает исходный размер."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def add_noise_and_tone(
    image: np.ndarray,
    rng: np.random.Generator,
    alpha: tuple[float, float],
    beta: tuple[float, float],
    noise: tuple[float, float],
) -> np.ndarray:
    """Меняет яркость и контраст, затем добавляет гауссов шум."""
    result = image.astype(np.float32) * rng.uniform(*alpha) + rng.uniform(*beta)
    result += rng.normal(0.0, rng.uniform(*noise), image.shape)
    return np.clip(result, 0, 255).astype(np.uint8)


def degrade(image: np.ndarray, rng: np.random.Generator, level: str) -> np.ndarray:
    """Применяет искажения заданной сложности в фиксированном порядке."""
    if level == "moderate":
        image = resize_roundtrip(image, rng.uniform(0.55, 0.82))
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.45, 1.0))
        if rng.random() < 0.55:
            image = motion_blur(image, int(rng.choice([3, 5])), rng)
        image = add_noise_and_tone(image, rng, (0.82, 1.1), (-12, 10), (1, 6))
        return jpeg_roundtrip(image, int(rng.integers(38, 71)))
    if level == "hard_readable":
        image = resize_roundtrip(image, rng.uniform(0.32, 0.58))
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.8, 1.5))
        image = motion_blur(image, int(rng.choice([3, 5, 7])), rng)
        image = add_noise_and_tone(image, rng, (0.7, 1.03), (-18, 12), (2, 9))
        return jpeg_roundtrip(image, int(rng.integers(20, 46)))
    raise ValueError(level)


def motion_blur(image: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Размывает изображение случайно выбранным направленным ядром."""
    kernel = np.zeros((length, length), dtype=np.float32)
    if rng.random() < 0.7:
        kernel[length // 2, :] = 1.0
    elif rng.random() < 0.5:
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def structural_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Считает корреляцию яркости исходного и искажённого изображения."""
    first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    second = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def readable(
    reference: np.ndarray, candidate: np.ndarray, level: str
) -> tuple[bool, float]:
    """Проверяет, осталось ли искажённое изображение читаемым."""
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    correlation = structural_correlation(reference, candidate)
    minimum_correlation = 0.84 if level == "moderate" else 0.62
    minimum_std = max(12.8, 0.55 * float(original_gray.std()))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (
        correlation >= minimum_correlation
        and gray.std() >= minimum_std
        and (laplacian >= 20.0),
        correlation,
    )


def guarded_degrade(
    image: np.ndarray, rng: np.random.Generator, level: str
) -> tuple[np.ndarray, float]:
    """Повторяет искажение до получения читаемого результата."""
    best_image, best_correlation = (image, 1.0)
    for _ in range(16):
        candidate = degrade(image, rng, level)
        accepted, correlation = readable(image, candidate, level)
        if correlation < best_correlation:
            best_image, best_correlation = (candidate, correlation)
        if accepted:
            return (candidate, correlation)
    if level == "hard_readable":
        return guarded_degrade(image, rng, "moderate")
    return (best_image, best_correlation)


def curve_upright(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Изгибает горизонтальную строку дугой, сохраняя широкий формат."""
    height, width = image.shape[:2]
    amplitude = rng.uniform(0.24, 0.4) * height
    padding = int(np.ceil(amplitude)) + 2
    output_height = height + 2 * padding
    x = np.arange(width, dtype=np.float32)
    normalized_x = 2.0 * x / max(1, width - 1) - 1.0
    shift = -amplitude * (1.0 - normalized_x**2)
    map_x = np.broadcast_to(x, (output_height, width)).copy()
    output_y = np.arange(output_height, dtype=np.float32)[:, None]
    map_y = np.broadcast_to(
        output_y - padding - shift[None, :], (output_height, width)
    ).copy()
    border = tuple((float(value) for value in np.median(image.reshape(-1, 3), axis=0)))
    curved = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    if curved.shape[1] / curved.shape[0] < 6.0:
        curved = cv2.resize(
            curved,
            (int(np.ceil(6.05 * curved.shape[0])), curved.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return curved


def image_stats(image: np.ndarray) -> dict[str, float | int]:
    """Считает размеры, контраст и резкость изображения."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "ratio": image.shape[1] / image.shape[0],
        "std": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def make_contact_sheet(rows: list[dict[str, str]], output: Path) -> None:
    """Создаёт обзорный лист с примерами каждого вида искажений."""
    examples = []
    groups = [
        (
            "clean",
            lambda row: row["degradation"] == "clean" and row["geometry"] == "straight",
        ),
        ("moderate", lambda row: row["degradation"] == "moderate"),
        ("hard_readable", lambda row: row["degradation"] == "hard_readable"),
        ("curved", lambda row: row["geometry"] == "curved"),
    ]
    for name, predicate in groups:
        for row in [item for item in rows if predicate(item)][:8]:
            image = cv2.imread(row["image_path"])
            height = 82
            width = min(480, max(120, round(image.shape[1] * height / image.shape[0])))
            resized = cv2.resize(image, (width, height))
            canvas = np.full((108, 500, 3), 245, dtype=np.uint8)
            canvas[:height, :width] = resized
            cv2.putText(
                canvas,
                f"{name}: {row['image_id']}",
                (5, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            examples.append(canvas)
    cv2.imwrite(str(output), np.vstack(examples))


def main() -> None:
    """Отбирает строки, воспроизводимо искажает их и сохраняет датасет."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/train/yandex_ocr_orientation_eval/labels.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/train/yandex_ocr_orientation_eval_ratio_ge_6_realistic_v2"),
    )
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    with args.labels.open(newline="", encoding="utf-8") as file:
        candidates = []
        for row in csv.DictReader(file):
            if float(row["width"]) / float(row["height"]) < 6.0:
                continue
            image = cv2.imread(row["image_path"])
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[0] >= 16 and image.shape[1] >= 100 and (gray.std() >= 12.8):
                candidates.append((row, image))
    order = list(range(len(candidates)))
    random.Random(args.seed).shuffle(order)
    hard = set(order[: round(0.1 * len(order))])
    moderate = set(order[round(0.1 * len(order)) : round(0.35 * len(order))])
    curve_pool = [
        index
        for index in order
        if candidates[index][1].shape[1] / candidates[index][1].shape[0] >= 8.0
    ]
    curved = set(curve_pool[: round(0.12 * len(candidates))])
    images_output = args.output / "images"
    images_output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (source, original) in enumerate(candidates):
        rng = np.random.default_rng(args.seed + index)
        degradation = "clean"
        correlation = 1.0
        image = original
        if index in hard:
            degradation = "hard_readable"
            image, correlation = guarded_degrade(image, rng, degradation)
        elif index in moderate:
            degradation = "moderate"
            image, correlation = guarded_degrade(image, rng, degradation)
        geometry = "straight"
        if index in curved:
            geometry = "curved"
            target = int(source["target"])
            upright = cv2.rotate(image, cv2.ROTATE_180) if target else image
            upright = curve_upright(upright, rng)
            image = cv2.rotate(upright, cv2.ROTATE_180) if target else upright
        stats = image_stats(image)
        if stats["ratio"] < 6.0:
            geometry = "straight"
            image = (
                original
                if degradation == "clean"
                else guarded_degrade(original, rng, degradation)[0]
            )
            stats = image_stats(image)
        output_path = images_output / f"{source['image_id']}.png"
        cv2.imwrite(str(output_path), image)
        row = dict(source)
        row.update(
            {
                "image_path": str(output_path),
                "degradation": degradation,
                "geometry": geometry,
                "structural_correlation": f"{correlation:.6f}",
                "output_width": str(stats["width"]),
                "output_height": str(stats["height"]),
                "output_ratio": f"{stats['ratio']:.6f}",
                "output_std": f"{stats['std']:.6f}",
                "output_laplacian_variance": f"{stats['laplacian_variance']:.6f}",
            }
        )
        rows.append(row)
    manifest = args.output / "labels.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(rows, args.output / "contact_sheet.jpg")
    summary = {
        "samples": len(rows),
        "degradation": {
            name: sum((row["degradation"] == name for row in rows))
            for name in ["clean", "moderate", "hard_readable"]
        },
        "geometry": {
            name: sum((row["geometry"] == name for row in rows))
            for name in ["straight", "curved"]
        },
        "selection": "ratio>=6, height>=16, width>=100, grayscale_std>=12.8",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
