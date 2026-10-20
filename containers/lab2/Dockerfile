FROM python:3.11-slim
#Метаданные
LABEL maintainer="yamalieva@lab.local"
LABEL version="1.0"

WORKDIR /app

#файл зависимостей
COPY requirements.txt .

#зависимости Python
RUN pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 5436

#Запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5436"]