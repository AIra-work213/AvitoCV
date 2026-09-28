# Как воспроизвести финальный сабмит

Финальный файл:
[`submission_ocr_v5_meta_yandex_selected_b8.csv`](../submission_ocr_v5_meta_yandex_selected_b8.csv).

В нём ровно 20 000 строк и две колонки:

- `image_id` — имя конкурсного изображения;
- `p_180` — вероятность того, что текст повёрнут на 180°.

Результат этого файла на лидерборде:

```text
1 - Brier Score = 0.99208426
Brier Score     = 0.00791574
```

## Из чего состоит решение

У решения два уровня.

Первый уровень — базовый ансамбль:

- PP-LCNet x0.25 для определения ориентации;
- PP-LCNet x1.0 для определения ориентации;
- RapidOCR orientation classifier;
- multi-crop обработка очень длинных строк.

Второй уровень — небольшая логистическая регрессия. Она получает базовую
вероятность и признаки от трёх OCR-распознавателей:

- `cyrillic_PP-OCRv5_mobile_rec`;
- `eslav_PP-OCRv5_mobile_rec`;
- `en_PP-OCRv5_mobile_rec`.

Каждый OCR запускается на исходном изображении и на его копии, повёрнутой на
180°. Правильная ориентация обычно даёт более уверенное и более правдоподобное
распознавание текста.

## 1. Установка окружения

Требуется Python 3.12 и менеджер окружений `uv`.

Из корня проекта выполните:

```bash
uv sync --frozen
```

Список библиотек находится в [`pyproject.toml`](../pyproject.toml), а точные
версии зафиксированы в [`uv.lock`](../uv.lock).

## 2. Размещение конкурсных данных

После распаковки конкурсного архива структура должна выглядеть так:

```text
data/test/
├── sample_submission.csv
└── test/
    └── images/
        ├── test_00000.png
        ├── test_00001.png
        └── ...
```

В `data/test/test/images` должно быть ровно 20 000 PNG-файлов. Имя каждого
файла без расширения должно совпадать с `image_id` в `sample_submission.csv`.

## 3. Какие готовые файлы использует финальная сборка

Чтобы не прогонять тяжёлые модели при каждом запуске, их результаты сохранены в
CSV.

### Базовая вероятность

[`submission_three_model_two_regime_tuned.csv`](../submission_three_model_two_regime_tuned.csv)
содержит вероятность базового ансамбля PP-LCNet x0.25, PP-LCNet x1.0 и
RapidOCR, включая multi-crop режим для длинных строк.

### Три файла с результатами OCR

1. [`outputs/test_cyrillic_v5_rec_predictions_b8.csv`](../outputs/test_cyrillic_v5_rec_predictions_b8.csv)
   — кириллическая модель.
2. [`outputs/test_eslav_v5_rec_predictions_b8.csv`](../outputs/test_eslav_v5_rec_predictions_b8.csv)
   — восточнославянская модель.
3. [`outputs/test_english_v5_rec_predictions_b8.csv`](../outputs/test_english_v5_rec_predictions_b8.csv)
   — английская модель.

У всех трёх файлов одинаковые колонки:

| Колонка | Значение |
| --- | --- |
| `image_id` | Идентификатор конкурсной картинки |
| `rec_score_direct` | Уверенность OCR в исходной ориентации |
| `rec_score_rotated` | Уверенность OCR после поворота на 180° |
| `rec_score_difference` | `rotated - direct` |
| `rec_text_direct` | Текст, распознанный в исходной ориентации |
| `rec_text_rotated` | Текст, распознанный после поворота |

Положительная `rec_score_difference` означает, что OCR увереннее распознаёт
повёрнутую версию.

Для обучения метамодели используются такие же файлы, рассчитанные на 553
Yandex-примерах. Про их создание подробно написано в
[`YANDEX_VALIDATION_DATASET.md`](YANDEX_VALIDATION_DATASET.md).

## 4. Быстрое точное воспроизведение

Если готовые CSV-кэши находятся на своих местах, достаточно одной команды:

```bash
uv run python scripts/create_submission_ocr_v5_meta.py
```

Скрипт
[`scripts/create_submission_ocr_v5_meta.py`](../scripts/create_submission_ocr_v5_meta.py)
выполнит следующие действия:

1. Загрузит базовые вероятности и результаты трёх OCR-моделей.
2. Обучит символьную языковую модель на расшифровках Yandex OCR-датасета.
3. Рассчитает признаки уверенности, качества распознанного текста, разногласия
   OCR-моделей, отношения сторон и базовой вероятности.
4. Обучит `StandardScaler + LogisticRegression` на 553 Yandex-примерах.
5. Применит фиксированную температурную калибровку `T=0.42`.
6. Запишет сабмит и JSON с параметрами запуска.

Параметры финальной метамодели:

```text
C = 0.3
class_weight = balanced
random_state = 20260927
temperature = 0.42
```

На выходе появятся:

```text
submission_ocr_v5_meta_yandex_selected_b8.csv
submission_ocr_v5_meta_yandex_selected_b8.json
```

## 5. Проверка точного совпадения

```bash
sha256sum submission_ocr_v5_meta_yandex_selected_b8.csv
```

Ожидаемая контрольная сумма:

```text
9580d7080b3e19c0b3e01e80e2c291743739caa2964a6dd049bbb9cc99226ffd
```

Если SHA-256 совпал, получившийся CSV побайтово равен отправленному файлу.

## 6. Как заново получить три OCR-файла

За расчёт отвечает
[`scripts/cache_test_recognizer_orientation.py`](../scripts/cache_test_recognizer_orientation.py).

Скрипт поддерживает продолжение после остановки: если выходной CSV уже
существует, ранее обработанные `image_id` будут пропущены. Для полностью нового
прогона используйте пустой каталог или временно переместите существующие CSV.

### Кириллическая модель

```bash
uv run python scripts/cache_test_recognizer_orientation.py \
  --model-name cyrillic_PP-OCRv5_mobile_rec \
  --images data/test/test/images \
  --sample data/test/sample_submission.csv \
  --output outputs/test_cyrillic_v5_rec_predictions_b8.csv \
  --batch-size 8
```

### Восточнославянская модель

```bash
uv run python scripts/cache_test_recognizer_orientation.py \
  --model-name eslav_PP-OCRv5_mobile_rec \
  --images data/test/test/images \
  --sample data/test/sample_submission.csv \
  --output outputs/test_eslav_v5_rec_predictions_b8.csv \
  --batch-size 8
```

### Английская модель

```bash
uv run python scripts/cache_test_recognizer_orientation.py \
  --model-name en_PP-OCRv5_mobile_rec \
  --images data/test/test/images \
  --sample data/test/sample_submission.csv \
  --output outputs/test_english_v5_rec_predictions_b8.csv \
  --batch-size 8
```

Если моделей ещё нет в локальном кэше PaddleX, они будут скачаны автоматически.
Финальный эксперимент проводился с `batch_size=8`.

После завершения в каждом файле должно быть 20 000 уникальных `image_id`. Затем
можно снова запускать финальную сборку:

```bash
uv run python scripts/create_submission_ocr_v5_meta.py
```

## 7. Проверка формата сабмита

```bash
uv run python - <<'PY'
import pandas as pd

submission = pd.read_csv("submission_ocr_v5_meta_yandex_selected_b8.csv")

assert list(submission.columns) == ["image_id", "p_180"]
assert len(submission) == 20_000
assert submission["image_id"].nunique() == 20_000
assert submission["p_180"].between(0.0, 1.0).all()

print(submission["p_180"].describe())
PY
```

Значение лидерборда локально пересчитать нельзя, потому что правильные ответы
для конкурсного теста скрыты.

## Что именно воспроизводится

Точный финальный сабмит воспроизводится из конкурсных изображений и сохранённых
CSV с предсказаниями. Три PP-OCRv5 файла можно полностью пересчитать командами
выше. Yandex-выборка и её OCR-признаки также собираются заново по инструкции из
[`YANDEX_VALIDATION_DATASET.md`](YANDEX_VALIDATION_DATASET.md).

Базовый тестовый ансамбль и сырые multi-crop предсказания Yandex сохранены как
входные артефакты финального решения. После очистки минимального репозитория в
нём нет полного набора исторических скриптов, позволяющего восстановить именно
эти два файла только из изображений. Удалённые файлы доступны через Git-тег
`recovery/pre-final-minimal-20260927`.
