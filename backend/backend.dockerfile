# backend/Dockerfile
FROM python:3.10

RUN apt-get update && apt-get install -y poppler-utils
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENTRYPOINT ["python", "main.py"]