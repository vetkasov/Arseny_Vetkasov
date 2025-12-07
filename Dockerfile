# Базовый образ Python
FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка всех необходимых пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всех файлов проекта в контейнер
COPY . .
RUN mkdir -p /app/result
# Команда запуска вашего решения
CMD ["python", "main.py"]
