# data/

- `fotos/dataset/`: fotos del docente (una por persona; el id sale del nombre del fichero,
  p. ej. `aitana_etxebarria_(twin_melody)_2024.jpg` -> `aitana_etxebarria`). Son **49**: la persona 50 del dataset es la nuestra.
- `fotos/vinicius_junior/`: nuestras fotos de Vinicius Junior (5). Se inscriben todas.
- `ataques/gemela_paula/`: 2 fotos de Paula Etxeberria, gemela de Aitana. Solo para `scripts/ataque_gemela.py`; no se inscriben ni entran en la imagen Docker.
- `db.csv`: BD de templates cifrados (se genera con `python scripts/build_db.py`; no se versiona).
  Su clave y su huella viven en `keys/` (tampoco se versiona).
