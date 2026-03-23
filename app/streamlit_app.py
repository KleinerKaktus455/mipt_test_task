import base64
import io

import streamlit as st

try:
    from .pipeline import process_document_image
except ImportError:
    # Streamlit may execute this file as a script (no package context).
    from pipeline import process_document_image


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
            st.image(io.BytesIO(img_bytes), use_column_width=True)
        else:
            st.warning("Не удалось получить аннотированное изображение.")

    with col_fields:
        st.subheader("Извлечённые поля")
        fields = result.get("structured_fields", {})
        st.write(f"**ФИО:** {fields.get('full_name') or '—'}")
        st.write(f"**Дата рождения:** {fields.get('date_of_birth') or '—'}")
        st.write(f"**Номер документа:** {fields.get('document_number') or '—'}")

        st.subheader("Угол поворота")
        st.write(result.get("angle"))

    st.subheader("Полный JSON-ответ")
    st.json(result)


if __name__ == "__main__":
    main()

