import streamlit as st
import requests
from PIL import Image
import time
import os

def download_file_from_google_drive(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

logo = Image.open('images/logo_13.jpeg')
st.image(logo, width=800)

st.title("Приложения FastAPI и Streamlit")

# Размытие лиц
st.header("Размытие лиц")
st.write("Модель yolo11m.pt. Набор данных для обнаружения лиц.")
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

# Скачивание модели размытия лиц
yolo_url = "https://drive.google.com/uc?id=1XgsZ02qC8rks12K_1mW3RJFicPejs3_X"
yolo_path = "api/weights/face_yolo11m.pt"
download_file_from_google_drive(yolo_url, yolo_path)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение.", use_container_width=True)
    if st.button("Размыть лица"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/detect_face", files=files)
        if response.status_code == 200:
            st.image(response.content, caption="Изображение с размытыми лицами.", use_container_width=True)
        else:
            st.write("Ошибка при размытии лиц.")

# Классификация отзывов
st.header("Классификация отзывов на фильмы (BERT)")
st.write("Модель ruBert. Классификация отзывов на фильм.")

# Скачивание модели классификатора
model_url = "https://drive.google.com/uc?id=1QvOOq3n3KlUMDyJq4u2BMH0QyC8WKQBb"
model_path = "api/weights/bert_classifier/model.safetensors"
download_file_from_google_drive(model_url, model_path)

# Скачивание токенизатора
tokenizer_url = "https://drive.google.com/uc?id=1PDVoIapQ47GDjhnoVwcA_qkAcD-ckkpo"
tokenizer_path = "api/weights/bert_tokenizer/tokenizer.json"
download_file_from_google_drive(tokenizer_url, tokenizer_path)

review = st.text_area("Введите отзыв")

if st.button("Классифицировать отзыв (BERT)"):
    if review.strip() == "":
        st.error("Пожалуйста, введите отзыв.")
    else:
        start_time = time.time()
        response = requests.post("http://127.0.0.1:8000/sentiment", data={"text": review})
        bert_time = time.time() - start_time
        if response.status_code == 200:
            result = response.json()["sentiment"]
            st.write(f"Отзыв классифицирован как: **{result}**")
            st.write(f"Время предсказания: {bert_time:.4f} сек")
        else:
            st.write("Ошибка при классификации отзыва.")