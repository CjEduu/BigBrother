FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/download_models.py
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "bigbrother"]
