import base64
import io

import streamlit as st

try:
    from .pipeline import process_document_image
except ImportError:
    # Streamlit may execute this file as a script (no package context).
    from pipeline import process_document_image

# Подписи для полей RussianDocsOCR (неизвестные ключи показываются как есть).
FIELD_LABELS_RU: dict[str, str] = {
    "Last_name_ru": "Фамилия",
    "First_name_ru": "Имя",
    "Middle_name_ru": "Отчество",
    "Birth_date": "Дата рождения",
    "Issue_date": "Дата выдачи",
    "Expiration_date": "Дата окончания",
    "Licence_number": "Номер документа / ВУ",
    "Issue_organisation_code": "Код подразделения",
    "Birth_place_ru": "Место рождения (RU)",
    "Birth_place_en": "Место рождения (EN)",
    "Issue_organization_ru": "Кем выдан (RU)",
    "Issue_organization_en": "Кем выдан (EN)",
    "Living_region_ru": "Регион (RU)",
    "Living_region_en": "Регион (EN)",
    "Sex_ru": "Пол (RU)",
    "Sex_en": "Пол (EN)",
    "Last_name_en": "Фамилия (EN)",
    "First_name_en": "Имя (EN)",
    "Middle_name_en": "Отчество (EN)",
    "Driver_class": "Категории ВУ",
}


def _field_label(field_id: str) -> str:
    return FIELD_LABELS_RU.get(field_id, field_id.replace("_", " "))


def main() -> None:
    st.set_page_config(page_title="Документ OCR", layout="wide")
    st.title("Обработка документов (выравнивание + OCR)")
    st.write(
        "Загрузите изображение банковской карты, ID-карты или "
        "водительского удостоверения. Система выровняет изображение, "
        "распознает текст и попытается извлечь ФИО, дату рождения и номер документа."
    )

    uploaded = st.file_uploader("Изображение документа", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        return

    bytes_data = uploaded.read()

    with st.spinner("Обрабатываем изображение..."):
        result = process_document_image(bytes_data)

    col_img, col_fields = st.columns([2, 1])

    with col_img:
        st.subheader("Выровненное изображение с разметкой")
        img_b64 = result.get("annotated_image_base64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            st.image(io.BytesIO(img_bytes), use_container_width=True)
        else:
            st.warning("Не удалось получить аннотированное изображение.")

    with col_fields:
        st.subheader("Распознанные поля")
        recognized = result.get("recognized_fields")
        if isinstance(recognized, dict) and recognized:
            for field_id, value in recognized.items():
                label = _field_label(str(field_id))
                display = value if value else "—"
                st.markdown(f"**{label}**  \n{display}")
        else:
            st.info(
                "Поля OCR не получены (часто если тип документа не определён или OCR не выполнялся)."
            )

        st.subheader("Угол поворота")
        st.write(f"{result.get('angle', 0):.2f}°")

    st.subheader("Полный JSON-ответ")
    st.json(result)


if __name__ == "__main__":
    main()

