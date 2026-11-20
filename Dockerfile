FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bigbrother/ bigbrother/
COPY scripts/ scripts/
COPY models/ models/
COPY data/fotos/ data/fotos/
COPY tests/ tests/
COPY pytest.ini .

# Si el modelo YuNet no viene en el repo se descarga; despues se construye la BD con las
# identidades del dataset. Cada imagen construida tiene su propia clave maestra (keys/master.key).
RUN python scripts/download_models.py \
 && python scripts/build_db.py \
 && chmod 700 keys && chmod 600 keys/*

ENTRYPOINT ["python", "-m", "bigbrother"]
CMD ["--help"]
