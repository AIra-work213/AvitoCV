# Как был собран финальный Yandex-датасет

Этот датасет использовался для выбора конфигурации финальной метамодели и её
обучения. В нём 553 изображения текстовых строк. Содержимое строк взято из
реальных фотографий, а правильная ориентация создана автоматически: половина
кропов оставлена как есть, половина повёрнута на 180°.

Главная идея — получить выборку, которая похожа на конкурсный тест по форме и
качеству изображений, но имеет точно известные ответы.

## Короткий сценарий запуска

Все команды ниже выполняются из корня проекта.

Сначала установите зависимости:

```bash
uv sync --frozen
```

Затем обязательно восстановите OpenCV 5 последним установленным вариантом и
проверьте импортируемую версию:

```bash
uv pip install --reinstall --no-deps opencv-python==5.0.0.93
uv run python -c "import cv2; assert cv2.__version__ == '5.0.0'; print(cv2.__version__)"
```

OpenCV 4.10 создаёт немного другие пиксели после `warpPerspective` и не
подходит для точного воспроизведения сохранённых признаков. Скрипт подготовки
завершится ошибкой, если загружена другая версия.

Если выборка уже была создана под OpenCV 4.10, сначала сохраните ошибочные
результаты отдельно, чтобы их можно было восстановить:

```bash
mkdir -p recovery/opencv_4_10
mv data/train/yandex_ocr_real_long_lines_v4 recovery/opencv_4_10/
mv outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv recovery/opencv_4_10/
mv outputs/yandex_real_long_v4_multicrop_raw.csv recovery/opencv_4_10/
mv outputs/yandex_cyrillic_v5_rec_predictions.csv recovery/opencv_4_10/
mv outputs/yandex_eslav_v5_rec_predictions.csv recovery/opencv_4_10/
mv outputs/yandex_english_v5_rec_predictions.csv recovery/opencv_4_10/
```

После этого повторите подготовку выборки и расчёт всех пяти файлов признаков по
командам ниже. Генератор запишет в `summary.json` версию OpenCV и общую
контрольную сумму 553 изображений. Ожидаемая сумма:

```text
dc1cdaa3b4e97cdf5bf396faafb9c115b2676642e603ef29c52692e0e62bc636
```

Затем скачайте исходный Yandex-датасет:

```bash
uv run python scripts/download_yandex_ocr_wild.py
```

Сразу после завершения загрузки соберите финальную выборку из 553 строк:

```bash
uv run python scripts/prepare_yandex_ocr_real_lines_validation.py \
  --source data/train/rus_ocr_in_the_wild_dataset \
  --test-images data/test/test/images \
  --output data/train/yandex_ocr_real_long_lines_v4 \
  --samples 553 \
  --seed 20260926
```

Для этой команды уже должны лежать конкурсные изображения в
`data/test/test/images`. Они нужны для подбора похожего распределения отношения
ширины к высоте.

После успешной сборки проверьте наличие основных файлов:

```bash
test -f data/train/yandex_ocr_real_long_lines_v4/labels.csv
test -f data/train/yandex_ocr_real_long_lines_v4/summary.json
test -d data/train/yandex_ocr_real_long_lines_v4/images
```

После этого переходите к разделам
«Предсказания базового ансамбля» и «Предсказания трёх OCR-моделей»: там
приведены команды, которые создают признаки для обучения финальной модели.

## 1. Исходные данные

В качестве источника используется открытый набор
`rus_ocr_in_the_wild_dataset` из репозитория Yandex Cloud. Чтобы результат не
изменился при обновлении репозитория, зафиксирован конкретный коммит:

```text
6abc7e2c74aa4c9175236e3de14f15fd9785064b
```

Скрипт загрузки:
[`scripts/download_yandex_ocr_wild.py`](../scripts/download_yandex_ocr_wild.py).

Запуск из корня проекта:

```bash
uv run python scripts/download_yandex_ocr_wild.py
```

Скрипт скачивает 297 изображений и 297 файлов с разметкой. Для каждого файла
проверяется исходный Git SHA-1, поэтому повреждённая или изменившаяся копия не
будет принята незаметно.

Результат появится в каталоге
[`data/train/rus_ocr_in_the_wild_dataset`](../data/train/rus_ocr_in_the_wild_dataset).
Полный список исходных файлов и их контрольных сумм сохранён в
[`source_manifest.json`](../data/train/rus_ocr_in_the_wild_dataset/source_manifest.json).

### Что делать сразу после загрузки

Убедитесь, что конкурсные изображения находятся в
`data/test/test/images`, и запустите подготовку 553 примеров:

```bash
uv run python scripts/prepare_yandex_ocr_real_lines_validation.py \
  --source data/train/rus_ocr_in_the_wild_dataset \
  --test-images data/test/test/images \
  --output data/train/yandex_ocr_real_long_lines_v4 \
  --samples 553 \
  --seed 20260926
```

Именно эта команда читает скачанные фотографии и разметку, находит длинные
строки, создаёт варианты качества, поворачивает половину изображений и записывает
готовую выборку в `data/train/yandex_ocr_real_long_lines_v4`.

## 2. Как из фотографий получаются длинные строки

Основной скрипт подготовки:
[`scripts/prepare_yandex_ocr_real_lines_validation.py`](../scripts/prepare_yandex_ocr_real_lines_validation.py).

Вспомогательные преобразования качества и геометрии находятся в
[`scripts/prepare_yandex_ocr_realistic_validation.py`](../scripts/prepare_yandex_ocr_realistic_validation.py).

### Фильтрация исходных боксов

Из Yandex-разметки удаляются:

- пустые строки;
- области с текстом `###`;
- вертикальные боксы;
- боксы высотой меньше 3 пикселей.

Для каждого оставшегося бокса вычисляются его направление, ширина, высота и
центр.

### Поиск слов одной строки

Соседние боксы объединяются, если они, скорее всего, относятся к одной
физической строке текста:

- разница направления не превышает 12°;
- высота боксов отличается не более чем в 1,8 раза;
- расстояние по горизонтали не слишком велико;
- вертикальное смещение ограничено относительно высоты текста.

В группе должно быть не менее двух боксов. После этого из исходной фотографии
вырезается один непрерывный фрагмент и выравнивается перспективным
преобразованием. Отдельные картинки со словами друг с другом не склеиваются.

### Проверка полученного кропа

Кроп отбрасывается, если:

- алгоритм случайно объединил несколько рядов текста;
- отношение ширины к высоте меньше 6;
- высота меньше 12 пикселей;
- изображение почти однотонное и текст плохо различим.

После этих проверок остаётся 773 кандидата.

## 3. Почему выбраны именно 553 примера

В конкурсном тесте измеряется `width / height` для всех изображений с
отношением сторон не меньше 6. Из этого распределения берутся квантили, а затем
для каждого квантиля выбирается наиболее близкая по форме Yandex-строка.

Так отбираются 553 изображения, похожие на длинные конкурсные строки. Текст и
ориентация тестовых изображений при этом не анализируются.

Итоговое распределение:

| Условие | Количество изображений |
| --- | ---: |
| `ratio >= 6` | 553 |
| `ratio >= 7` | 440 |
| `ratio >= 8` | 359 |
| `ratio >= 9` | 273 |
| `ratio >= 10` | 203 |
| `ratio >= 15` | 62 |
| `ratio >= 20` | 23 |

Медианное отношение сторон равно примерно 8,94, максимальное — 40,42.

## 4. Имитация качества конкурсных изображений

Часть строк намеренно ухудшается:

- около 25% получают умеренное ухудшение;
- около 10% получают сильное, но текст должен оставаться читаемым;
- остальные сохраняются чистыми.

После каждого сильного преобразования проверяются структурное сходство с
оригиналом и контраст. Если структурная корреляция меньше `0.65` или текст почти
исчез, используется исходный чистый кроп.

Фактически получилось:

| Вариант | Количество |
| --- | ---: |
| Чистые изображения | 365 |
| Умеренно ухудшенные | 129 |
| Сильно ухудшенные, но читаемые | 57 |
| Возврат к чистому оригиналу | 2 |

Для строк с `ratio >= 8` с вероятностью 12% применяется искривление текста. Это
нужно для имитации надписей, идущих по слабой дуге. В готовой выборке 507 прямых
и 46 искривлённых строк.

## 5. Создание правильных ответов

Ровно половина готовых изображений поворачивается на 180°:

```text
target = 0  — изображение оставлено в исходной ориентации
target = 1  — изображение повёрнуто на 180°
```

Все случайные операции воспроизводимы:

- основной `seed`: `20260926`;
- выбор повёрнутой половины: `20260927`, то есть `seed + 1`.

## 6. Что должно получиться после сборки

После выполнения команды подготовки появятся:

- [`labels.csv`](../data/train/yandex_ocr_real_long_lines_v4/labels.csv) — метки,
  исходные фотографии, тип ухудшения, геометрия и статистики изображений;
- `images/yandex_real_line_*.png` — 553 готовых кропа;
- [`summary.json`](../data/train/yandex_ocr_real_long_lines_v4/summary.json) —
  сводная статистика;
- `contact_sheet.jpg` — лист для визуальной проверки.

## 7. Предсказания базового ансамбля

Для каждого примера рассчитываются сигналы PP-LCNet x0.25, PP-LCNet x1.0 и
RapidOCR. За это отвечает
[`scripts/evaluate_three_model_orientation.py`](../scripts/evaluate_three_model_orientation.py).

Если в проекте уже есть готовый файл
`outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv`, этот этап можно
пропустить. Он нужен только для пересчёта базовых предсказаний.

При пересчёте скрипт сначала ищет локальные модели в каталогах:

```text
models/PP-LCNet_x0_25_textline_ori_infer/
models/PP-LCNet_x1_0_textline_ori_infer/
```

Если каталогов нет или в них отсутствует хотя бы один файл модели, PaddleX
автоматически скачает официальные `PP-LCNet_x0_25_textline_ori` и
`PP-LCNet_x1_0_textline_ori`. Поэтому при первом запуске без локальных моделей
нужен интернет.

```bash
uv run python scripts/evaluate_three_model_orientation.py \
  --labels data/train/yandex_ocr_real_long_lines_v4/labels.csv \
  --output outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv \
  --batch-size 32
```

При автоматической загрузке в начале лога появится сообщение
`PaddleX will download ... automatically`. После скачивания вычисление
продолжится без дополнительной команды.

### Multi-crop предсказания для широких строк

Финальная метамодель использует отдельные предсказания по левой, центральной и
правой части строк с `ratio >= 10`. Их создаёт восстановленный и проверенный
скрипт
[`scripts/create_yandex_multicrop_cache.py`](../scripts/create_yandex_multicrop_cache.py).

Запускайте его после расчёта базовых предсказаний из предыдущего шага:

```bash
uv run python scripts/create_yandex_multicrop_cache.py \
  --labels data/train/yandex_ocr_real_long_lines_v4/labels.csv \
  --baseline outputs/yandex_ocr_three_model_eval_real_long_lines_v4.csv \
  --output outputs/yandex_real_long_v4_multicrop_raw.csv \
  --threshold-ratio 10 \
  --batch-size 32
```

Для PP-LCNet берутся кропы шириной `2 × height`, а для RapidOCR — шириной
`4 × height`. В каждой модели обрабатываются левая, центральная и правая части.
PP-LCNet дополнительно использует исходный и повёрнутый на 180° варианты.

Результат:
[`outputs/yandex_real_long_v4_multicrop_raw.csv`](../outputs/yandex_real_long_v4_multicrop_raw.csv).
В нём должно быть 203 строки без учёта заголовка.

Для зафиксированного окружения эталонная SHA-256 этого файла:

```text
f54593539769d48cfa3bfcf2bff5e7a5583e71940d9f19daf6ac234fe2f8a997
```

## 8. Предсказания трёх OCR-моделей

Каждая OCR-модель распознаёт исходный кроп и тот же кроп после поворота на 180°.
Сохраняются оба текста, обе уверенности и разница уверенностей.

Для этих трёх файлов обязателен `batch_size=32`: именно с ним были созданы
признаки, на которых обучается финальная метамодель. Размер батча влияет на
результат распознавания PaddleOCR, поэтому `8` даст другие CSV и другой сабмит.

```bash
uv run python scripts/evaluate_cyrillic_rec_orientation.py \
  --labels data/train/yandex_ocr_real_long_lines_v4/labels.csv \
  --model-name cyrillic_PP-OCRv5_mobile_rec \
  --output outputs/yandex_cyrillic_v5_rec_predictions.csv \
  --batch-size 32

uv run python scripts/evaluate_cyrillic_rec_orientation.py \
  --labels data/train/yandex_ocr_real_long_lines_v4/labels.csv \
  --model-name eslav_PP-OCRv5_mobile_rec \
  --output outputs/yandex_eslav_v5_rec_predictions.csv \
  --batch-size 32

uv run python scripts/evaluate_cyrillic_rec_orientation.py \
  --labels data/train/yandex_ocr_real_long_lines_v4/labels.csv \
  --model-name en_PP-OCRv5_mobile_rec \
  --output outputs/yandex_english_v5_rec_predictions.csv \
  --batch-size 32
```

Итоговые файлы:

- [`outputs/yandex_cyrillic_v5_rec_predictions.csv`](../outputs/yandex_cyrillic_v5_rec_predictions.csv);
- [`outputs/yandex_eslav_v5_rec_predictions.csv`](../outputs/yandex_eslav_v5_rec_predictions.csv);
- [`outputs/yandex_english_v5_rec_predictions.csv`](../outputs/yandex_english_v5_rec_predictions.csv).

Контрольные суммы всех входов метамодели:

| Файл | SHA-256 |
| --- | --- |
| `labels.csv` | `aeadb9465297e359afe436c77091b47b2184b82ecf4912a8060f554bc9476fb6` |
| `yandex_ocr_three_model_eval_real_long_lines_v4.csv` | `76f629d5880146fcf543d49f3937b29b23943b4937b5a23ddce7dac574a968dc` |
| `yandex_real_long_v4_multicrop_raw.csv` | `f54593539769d48cfa3bfcf2bff5e7a5583e71940d9f19daf6ac234fe2f8a997` |
| `yandex_cyrillic_v5_rec_predictions.csv` | `95e44cd20877e8f22aac3338bb325b0642d1bd3dd84091ccc4f461fc1e6c59aa` |
| `yandex_eslav_v5_rec_predictions.csv` | `6970aeb597757db28f360bcba1b7a71c845b4e840e11de46b89b1300a814d968` |
| `yandex_english_v5_rec_predictions.csv` | `f90627b9d12474736353ec37cca0a6b13ee3fb611e5849504ed004f51b3759a4` |

Финальный сборщик проверяет эти суммы до обучения метамодели и не создаёт
сабмит, если хотя бы один вход отличается.

## 9. Как выборка применялась

В `labels.csv` для каждого кропа сохранено поле `source_image` — имя исходной
фотографии Yandex. При выборе признаков и параметров применялся пятифолдовый
`GroupKFold` с группировкой по `source_image`.

Поэтому разные фрагменты одной фотографии не могли одновременно попасть в
обучающую и проверочную части одного фолда. После выбора фиксированной
конфигурации финальная логистическая регрессия обучалась на всех 553 примерах.

## 10. Что делать после расчёта всех Yandex-файлов

Когда созданы `labels.csv`, предсказания базового ансамбля, multi-crop файл и
три OCR-файла, Yandex-часть подготовки закончена. Дальше нужно рассчитать три
аналогичных OCR-файла для конкурсного теста и запустить финальный сборщик.

Полная последовательность описана в документе
[`REPRODUCE_FINAL_SUBMISSION.md`](REPRODUCE_FINAL_SUBMISSION.md). Если тестовые
OCR-файлы уже готовы, сабмит собирается командой:

```bash
uv run python scripts/create_submission_ocr_v5_meta.py
```
