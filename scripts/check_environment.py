#!/usr/bin/env python3
"""Проверяет версии, от которых зависит побайтовая воспроизводимость."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

EXPECTED_PYTHON = (3, 12)
EXPECTED_OPENCV_PACKAGE = "5.0.0.93"
EXPECTED_OPENCV_CONTRIB_PACKAGE = "4.10.0.84"
EXPECTED_CV2 = "5.0.0"


def parse_args() -> argparse.Namespace:
    """Читает путь к отчёту о проверенном окружении."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/environment.json"),
    )
    return parser.parse_args()


def repair_opencv() -> None:
    """Переустанавливает OpenCV 5 после конфликтующего contrib-wheel."""
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--reinstall",
            "--no-deps",
            f"opencv-python=={EXPECTED_OPENCV_PACKAGE}",
        ],
        check=True,
    )


def validate_environment() -> dict[str, str]:
    """Проверяет Python и версии обоих пакетов OpenCV."""
    import cv2

    python_version = sys.version_info[:2]
    opencv_package = version("opencv-python")
    opencv_contrib_package = version("opencv-contrib-python")
    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            "Для точного воспроизведения требуется Python "
            f"{EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"загружен {python_version[0]}.{python_version[1]}"
        )
    if opencv_package != EXPECTED_OPENCV_PACKAGE:
        raise RuntimeError(
            f"Требуется opencv-python=={EXPECTED_OPENCV_PACKAGE}, "
            f"установлен {opencv_package}"
        )
    if opencv_contrib_package != EXPECTED_OPENCV_CONTRIB_PACKAGE:
        raise RuntimeError(
            "Требуется opencv-contrib-python=="
            f"{EXPECTED_OPENCV_CONTRIB_PACKAGE}, установлен {opencv_contrib_package}"
        )
    if cv2.__version__ != EXPECTED_CV2:
        raise RuntimeError(
            f"Требуется cv2=={EXPECTED_CV2}, загружен cv2=={cv2.__version__}"
        )
    return {
        "status": "ok",
        "python": f"{python_version[0]}.{python_version[1]}",
        "opencv_python": opencv_package,
        "opencv_contrib_python": opencv_contrib_package,
        "cv2": cv2.__version__,
    }


def main() -> None:
    """Проверяет окружение и сохраняет детерминированный JSON-отчёт."""
    args = parse_args()
    repair_opencv()
    report = validate_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
