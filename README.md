## Document OCR Service (FastAPI + Streamlit + RussianDocsOCR)

Сервис для обработки фото документов РФ:

- автообрезка документа в кадре (опционально),
- выравнивание (deskew),
- OCR через `RussianDocsOCR`,
- возврат структурированных полей и аннотированного изображения.

Проект теперь состоит из двух приложений:

- `api` (FastAPI) на `http://localhost:8000`,
- `streamlit` UI на `http://localhost:8501`.

---

### Текущая архитектура

- **`app/main.py`** - REST API:
  - `POST /process` - обработка изображения документа.
  - `GET /health` - health-check.
- **`app/pipeline.py`** - основной пайплайн:
  - предобработка и deskew,
  - интеграция с `RussianDocsOCR`,
  - fallback при `doctype=NONE`,
  - формирование JSON-ответа.
- **`app/streamlit_app.py`** - UI для ручной проверки результата OCR.
- **`docker-compose.yml`** - конфигурация для Linux/Windows (не macOS).
- **`docker-compose.macos.yml`** - конфигурация для macOS (CPU, `platform: linux/amd64`).

---

### Поддерживаемые документы (RussianDocsOCR)

По официальному README библиотеки `RussianDocsOCR`:

1. Внутренний паспорт РФ (версия 1997 года)
2. Внутренний паспорт РФ (версия 2011 года)
3. Заграничный паспорт РФ (версия 2003 года)
4. Заграничный паспорт РФ (биометрический, версия 2007 года)
5. Водительские права РФ (версия 2011 года)
6. Водительские права РФ (версия 2020 года)
7. СНИЛС (версия 1996 года)

---

### Как работает обработка

Пайплайн в `app/pipeline.py`:

1. Декодирование изображения из байтов.
2. Опциональная автообрезка по документу:
   - сначала через `RussianDocsOCR DocDetector`,
   - fallback на контуры OpenCV.
3. Выравнивание:
   - coarse-поворот на 90 град. (опционально),
   - fine deskew (projection/contour/hough/text),
   - дополнительная оценка угла через модули `RussianDocsOCR`.
4. OCR:
   - запуск `RussianDocsOCR Pipeline`,
   - если классификатор вернул `NONE`, возможно продолжение OCR с fallback doctype.
5. Формирование ответа:
   - `structured_fields`,
   - `recognized_fields`,
   - `raw_lines`, `detections`,
   - `annotated_image_base64`,
   - `alignment` (разложение итогового угла).

---

### API

#### `POST /process`

Принимает `multipart/form-data` с полем `file` (изображение).

Возвращает JSON с ключами:

- `angle` - итоговый примененный угол поворота.
- `alignment` - вклад каждого этапа:
  - `coarse_angle`,
  - `fine_angle`,
  - `russian_probe_angle90`,
  - `fields_angle`.
- `structured_fields`:
  - `full_name`,
  - `date_of_birth`,
  - `document_number`.
- `recognized_fields` - все OCR-поля словарем `field_id -> value`.
- `raw_lines` - текстовые строки OCR.
- `detections` - список боксов и confidence.
- `annotated_image_base64` - PNG в base64.
- `document_type` - тип документа от классификатора.
- `document_type_fallback` - присутствует, если сработал fallback.
- `quality` - данные по качеству из `RussianDocsOCR`.

#### `GET /health`

Простой health-check:

```json
{"status": "ok"}
```

---

### Локальный запуск (без Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Проверка:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Для UI (отдельным процессом):

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

---

### Docker: Linux/Windows (не macOS)

Используйте `docker-compose.yml`:

```bash
docker compose build
docker compose up
```

Сервисы:

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

По умолчанию режим CPU: `RUS_DOCS_DEVICE=cpu`.

Для GPU (при установленном NVIDIA Container Toolkit) включите `gpus: all` в обоих сервисах и задайте:

```bash
RUS_DOCS_DEVICE=gpu docker compose up
```

---

### Docker: macOS

Используйте отдельный файл `docker-compose.macos.yml`:

```bash
docker compose -f docker-compose.macos.yml build
docker compose -f docker-compose.macos.yml up
```

Особенности:

- принудительно `platform: linux/amd64`,
- CPU-only режим для `RussianDocsOCR`.

---

### Ключевые переменные окружения

Общие:

- `RUS_DOCS_DEVICE` - `cpu` или `gpu`.
- `RUS_DOCS_MODEL_FORMAT` - формат моделей (`ONNX` по умолчанию).
- `RUS_DOCS_IMG_SIZE` - размер длинной стороны для инференса (по умолчанию `1500`).
- `LLM_BASE_URL` - зарезервировано для возможной внешней LLM-интеграции.

Fallback для `doctype=NONE`:

- `RUS_DOCS_NONE_OCR_FALLBACK` - `1/0`, продолжать ли OCR при `NONE`.
- `RUS_DOCS_NONE_FALLBACK_DOCTYPE` - шаблон doctype для fallback (по умолчанию `dl_2011`).

Автообрезка:

- `PREP_AUTO_DOC_CROP` - включить автообрезку (`1/0`).
- `PREP_CROP_METHOD` - `auto`, `russian_doc`, `contour`.
- `PREP_CROP_MARGIN_FRAC`, `PREP_CROP_MIN_DOC_FRAC`, `PREP_CROP_MAX_DOC_FRAC`.

Deskew:

- `DESKEW_PRIMARY` - базовый метод (`projection` по умолчанию).
- `DESKEW_COARSE_90` - грубый поворот 0/90/180/270.
- `DESKEW_USE_RUS_DOCS_FIELDS` - использовать probe угла через модули RussianDocsOCR.

---

### Стек

- Python 3.11
- FastAPI + Uvicorn
- Streamlit
- OpenCV
- RussianDocsOCR
- Docker / Docker Compose

## Document Preprocessing & OCR – ML Engineer Test

**Цель**: система для первичной обработки персональных документов (банковские карты, ID карты, водительские удостоверения):

- **Выравнивание изображения**
- **Распознавание текста (OCR)**
- **Извлечение структурированных полей** (ФИО, дата рождения, номер документа)
- **REST API** с JSON-ответом и выровненным изображением с разметкой (в base64)

---

### Архитектура и принятые решения

- **Язык**: `Python 3.11`
- **Фреймворки**:
  - **FastAPI** – лёгкий REST API с удобной документацией (`/docs`)
  - **PyTorch + EasyOCR** – опенсорс OCR-модель, работающая и на CPU, и на GPU
  - **OpenCV** – классический CV для выравнивания документа и отрисовки разметки
- **Сборка**:
  - `Dockerfile` для образа приложения
  - `docker-compose.yml` для запуска сервиса c поддержкой GPU (при наличии)

#### Пайплайн обработки (`app/pipeline.py`)

1. **Загрузка изображения**
   - Принимаем байты файла, декодируем через `cv2.imdecode` в BGR-изображение.

2. **Выравнивание (deskew)**
   - Перевод в градации серого + блюр.
   - Бинаризация по Отсу, инверсия.
   - Поиск контура максимальной площади (предположительно – документ).
   - Оценка угла через `cv2.minAreaRect` и последующий поворот изображения.

3. **OCR**
   - Инициализация `easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())` один раз на процесс.
   - Работа и на CPU, и на GPU (решение автоматически выбирает режим в зависимости от наличия CUDA).
   - Получаем боксы, текст и confidence.

4. **Структурирование информации**
   - Формируем список строк (`raw_lines`).
   - Простейшие эвристики:
     - **Дата рождения** – регулярное выражение по формату `DD.MM.YYYY` / `DD-MM-YYYY` / `DD/MM/YYYY`.
     - **Номер документа** – поиск числового шаблона, похожего на паспорт/ID.
     - **ФИО** – первая непустая строка без цифр (часто ФИО пишут заглавными).
   - В реальной системе этот блок можно заменить на правила/ML-модель под конкретные форматы документов.

5. **Аннотация изображения**
   - Для каждого детекта рисуем прямоугольник и небольшую подпись (`text (conf)`).
   - Возвращаем изображение с разметкой как PNG в **base64** (`annotated_image_base64`) внутри JSON.

#### REST API (`app/main.py`)

- **`POST /process`**
  - Вход: `multipart/form-data`, поле `file` – изображение документа.
  - Выход: JSON:
    - `angle` – угол поворота (для контроля deskew).
    - `structured_fields`:
      - `full_name`
      - `date_of_birth`
      - `document_number`
    - `raw_lines` – сырые текстовые строки OCR.
    - `detections` – список `{bbox, text, confidence}`.
    - `annotated_image_base64` – выровненное изображение с разметкой (PNG в base64).

- **`GET /health`**
  - Простой health-check.

---

### Запуск локально (без Docker)

1. Установить зависимости (желательно в отдельное виртуальное окружение):

```bash
pip install -r requirements.txt
```

2. Запустить сервис:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Открыть документацию Swagger:

- Браузер: `http://localhost:8000/docs`
- Там можно отправить `POST /process` с изображением документа.

---

### Запуск через Docker / Docker Compose

#### Требования на целевой машине

- Ubuntu Server 22.04
- Docker + Docker Compose v2.25.0
- NVIDIA драйвер и (опционально) `nvidia-container-toolkit` для GPU

#### Сборка и запуск

```bash
docker compose build
docker compose up
```

Сервис будет доступен по адресу:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

---

### Поддержка GPU / CPU

- В `pipeline.py`:
  - `gpu=torch.cuda.is_available()` – EasyOCR автоматически использует GPU при наличии CUDA.
- В `docker-compose.yml`:
  - В секции `deploy.resources.reservations.devices` запрошен ресурс `gpu` (совместимо с современными Docker+NVIDIA).
  - При необходимости можно дополнительно указать `runtime: nvidia` (зависит от конфигурации).

Таким образом, одна и та же сборка работает:

- На **CPU** (когда CUDA недоступна).
- На **GPU** (например, NVIDIA GeForce RTX 2080 Ti на тестовом стенде).

---

### Настройка внешних LLM API (base_url / proxy)

Хотя текущее решение **не требует LLM** для основного пайплайна, добавлена переменная:

- `LLM_BASE_URL` – можно задать в `docker-compose.yml` или через окружение.

Если вы будете расширять сервис (например, пост-обработка OCR через LLM), можно:

- Прочитать `LLM_BASE_URL` в Python-клиенте (например, OpenAI / Anthropic) и использовать его как `base_url` или прокси-эндпоинт.

---

### Выравнивание (deskew) и RussianDocsOCR

Пайплайн (`app/pipeline.py`) может оценивать наклон **до** полного OCR: OpenCV (projection/contour) + геометрия **RussianDocsOCR** (сегментация документа и боксы полей).

#### Поддерживаемые документы 

1. Внутренний паспорт РФ (версия 1997 года)
2. Внутренний паспорт РФ (версия 2011 года)
3. Заграничный паспорт РФ (версия 2003 года)
4. Заграничный паспорт РФ (биометрический, версия 2007 года)
5. Водительские права РФ (версия 2011 года)
6. Водительские права РФ (версия 2020 года)
7. СНИЛС (версия 1996 года)

Полезные переменные окружения:

| Переменная | Значение по умолчанию | Смысл |
|------------|------------------------|--------|
| `DESKEW_USE_RUS_DOCS_FIELDS` | `1` | Включить пробу угла через RussianDocsOCR |
| `DESKEW_RUSDOC_PROBE_MODE` | `full` | `full` — два прохода (документ + поля); `doc_only` / `fields_only` — только один источник (быстрее или для отладки) |
| `DESKEW_DOC_FIELDS_MERGE_TOL` | `22` | Если угол по документу и по полям расходятся больше этого порога (градусы), берётся **только** угол по документу |
| `DESKEW_FIELDS_FORCE_MIN_BOXES` | `2` | Минимум боксов полей, чтобы считать угол (прямой вызов детекторов не зависит от `doctype`) |
| `DESKEW_DOC_FIELDS_ON_DISAGREE` | `doc` | При расхождении doc vs fields: `doc` или `fields` |
| `DESKEW_RUS_SKIP_ANGLE90` | `1` | `1` (по умолчанию) — не вызывать Angle90 в пробе, чтобы угол совпадал с кадром после OpenCV-deskew; `0` — как в библиотеке (тогда к кадру применяются те же шаги 90°) |
| `DESKEW_RUS_INVERT_ANGLE` | `0` | `1` — инвертировать знак угла (подбор при неверном направлении поворота) |
| `DESKEW_FIELDS_MIN_ANGLE` / `DESKEW_FIELDS_MAX_ANGLE` | `0.4` / `50` | Игнорировать слишком малый или подозрительно большой итоговый угол |
| `DESKEW_FIELDS_ROW_HULL_AGREE` | `14` | Согласованность двух эвристик по полям (строки vs оболочка боксов) |
| `DESKEW_PRIMARY` | `projection` | Основной классический deskew; см. также `DESKEW_AGREE_TOL`, `DESKEW_ON_DISAGREE_USE` в коде |

В ответе API поле `alignment` содержит только разложение **углов** (градусы): `coarse_angle`, `fine_angle`, `russian_probe_angle90`, `fields_angle`; суммарный поворот — в поле `angle`.

**Важно:** стандартный `Pipeline()` RussianDocsOCR при типе документа `NONE` **не доходит** до DocDetector/TextFieldsDetector — поэтому проба угла внутри пайплайна реализована через прямые вызовы модулей, иначе на «сложных» фото сегментация полей не использовалась бы вообще.

При полном OCR, если классификатор вернул `NONE`, наш сервис **всё равно** продолжает цепочку (детектор документа → поля → OCR), подставляя шаблон полей из переменной **`RUS_DOCS_NONE_FALLBACK_DOCTYPE`** (по умолчанию `dl_2011`, т.е. водительское удостоверение). В JSON тогда появляется **`document_type_fallback`**. Отключить: **`RUS_DOCS_NONE_OCR_FALLBACK=0`**. Для паспорта можно задать, например, `intpassport_2014` (формат `тип_год`, как в RussianDocsOCR).

#### Автообрезка под документ (как ручная подрезка фона)

Если документ на фото маленький, детекторам и OCR часто проще работать после **кадрирования**. Включите:

| Переменная | По умолчанию | Смысл |
|------------|----------------|--------|
| `PREP_AUTO_DOC_CROP` | `0` | `1` — перед deskew/OCR обрезать кадр по контуру документа |
| `PREP_CROP_METHOD` | `auto` | `auto` — сначала **DocDetector** (`russian_doc`), иначе контуры OpenCV (`contour`); можно задать `russian_doc` или `contour` явно |
| `PREP_CROP_MARGIN_FRAC` | `0.035` | Запас по краям от размера кадра (доля ширины/высоты) |
| `PREP_CROP_MIN_DOC_FRAC` | `0.08` | Мин. доля площади AABB документа на уменьшенном кадре (отсекаем ложные срабатывания) |
| `PREP_CROP_MAX_DOC_FRAC` | `0.93` | Макс. доля — если «документ» почти на весь кадр, обрезка не делается |
| `PREP_CROP_MAX_CROP_FRAC` | `0.985` | Если после полей обрезка почти на весь кадр — пропуск |

---

### Пример типичного ответа `/process`

```json
{
  "angle": -2.5,
  "structured_fields": {
    "full_name": "ИВАНОВ ИВАН ИВАНОВИЧ",
    "date_of_birth": "01.01.1990",
    "document_number": "1234 567890"
  },
  "raw_lines": [
    "РОССИЙСКАЯ ФЕДЕРАЦИЯ",
    "ИВАНОВ ИВАН ИВАНОВИЧ",
    "01.01.1990",
    "1234 567890"
  ],
  "detections": [
    {
      "bbox": [10, 20, 200, 60],
      "text": "ИВАНОВ ИВАН ИВАНОВИЧ",
      "confidence": 0.98
    }
  ],
  "annotated_image_base64": "<base64-строка PNG>"
}
```

---

### Использование LLM при разработке

При проектировании решения использовался агент-помощник (LLM) для:

- Быстрого подбора стека (PyTorch, EasyOCR, FastAPI, OpenCV).
- Чернового генератора кода пайплайна и REST API.
- Формирования структуры репозитория и README.

Далее код был вручную адаптирован под требования задания:

- Упор на локальный инференс (EasyOCR на PyTorch).
- Поддержка CPU/GPU.
- Docker + Docker Compose, ориентированные на Ubuntu Server 22.04 + RTX 2080 Ti.

