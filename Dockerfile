# Выбор базового образа с Python
FROM python:3.12-slim

# Установка рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY . /app

# Установка зависимостей
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Открытие порта для взаимодействия с приложением
EXPOSE 8080

# Запуск приложения
CMD ["python", "main.py"]