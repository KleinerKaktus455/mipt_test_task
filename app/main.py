from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline import process_document_image

app = FastAPI(
    title="Document Preprocessing & OCR API",
    description=(
        "Сервис для выравнивания изображения документа, распознавания текста "
        "и извлечения структурированных полей (ФИО, дата рождения, номер документа)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Принимает изображение документа, возвращает:
    - выравненный документ с разметкой (в base64 внутри JSON)
    - извлечённые структурированные поля
    - сырые строки и детекции OCR
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Ожидается файл изображения")

    try:
        content = await file.read()
        result = process_document_image(content)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {exc}") from exc

    return JSONResponse(result)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

