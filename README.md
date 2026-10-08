# BigBrother

Sistema de reconocimiento facial con **Multiscale Local Binary Patterns** (Practica de Sistemas Biometricos).

## Uso rapido
```bash
docker build -t bigbrother .
docker run --rm -v $(pwd)/data:/app/data bigbrother inscribir vinicius_junior /app/data/faces/vini1.jpg
docker run --rm -v $(pwd)/data:/app/data bigbrother identificar /app/data/faces/vini2.jpg
```

Local:
```bash
pip install -r requirements.txt
python scripts/download_models.py
pytest
```

API (`bigbrother.BigBrother`): `inscribir`, `verificar`, `identificar`, y las variantes `_sd` (sin detector) y `_t` (solo template).
Ver `docs/PLAN.md`.
