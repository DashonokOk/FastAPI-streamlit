# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn Pillow

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics transformers torch

# Копируем все файлы приложения
COPY . .

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Запускаем FastAPI приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]