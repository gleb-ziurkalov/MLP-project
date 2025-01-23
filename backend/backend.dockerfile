# backend/Dockerfile
FROM python:3.10


COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENTRYPOINT ["python", "main.py"]