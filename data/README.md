# data/

- `fotos/dataset/`: fotos del docente (una por persona; el id sale del nombre del fichero,
  p. ej. `aitana_etxebarria_(twin_melody)_2024.jpg` -> `aitana_etxebarria`). Recibimos **49**.
- `fotos/vinicius_junior/`: nuestras fotos de Vinicius Junior (5). Se inscriben todas.
- `db.csv`: BD de templates cifrados (se genera con `python scripts/build_db.py`; no se versiona).
  Su clave y su huella viven en `keys/` (tampoco se versiona).
