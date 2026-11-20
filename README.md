# BigBrother — reconocimiento facial con Multiscale LBP

Práctica de Sistemas Biométricos (UMA) — grupo **EVE**: Eduardo Jiliberto Cañizal, Alberto Bravo
Ramírez, Jorge Manuel Gutiérrez Melgar, Nacho Ferrer Tellez, Pablo Cervantes Pardo.

- Aproximación asignada: **Multiscale Local Binary Patterns (MLBP)**.
- Famoso elegido (persona 51): **Vinícius José Paixão de Oliveira Júnior** → id `vinicius_junior`.
- Informe de vulnerabilidades y fortalezas: **[docs/INFORME.md](docs/INFORME.md)**.

```
foto ─► YuNet (cara + 5 puntos + control de calidad) ─► alineado 128×128 ─► Tan-Triggs
     ─► MLBP: LBP u2 (P=8) radios 1,2,3,4 × rejilla 8×8 × 59 bins = TEMPLATE (15.104 floats)
     ─► √ ─► PCA blanqueado ─► LDA (entrenadas en LFW) ─► coseno ─► umbral
BD: data/db.csv  (id, nonce, template proyectado cifrado con AES-256-GCM)
```

## 1. Ejecutar con Docker (recomendado)

```bash
docker build -t bigbrother .
# volumen para que la BD persista entre ejecuciones (inscripciones nuevas)
docker volume create bb-data
alias bb='docker run --rm -v bb-data:/app/data -v "$PWD/pruebas:/pruebas" bigbrother'

bb listar                                          # identidades inscritas (49 + vinicius)
bb verificar vinicius_junior /pruebas/vini.jpg     # {"id": ..., "verificado": true|false}
bb identificar /pruebas/foto.jpg                   # {"identidad": "vinicius_junior" | null}
bb inscribir pepe /pruebas/pepe.jpg
```

Pon tus fotos en una carpeta `pruebas/` del directorio actual. En Windows PowerShell usa
`${PWD}` en lugar de `$PWD`. La imagen ya trae la BD construida con todas las identidades
y **su propia clave** (se genera en el `docker build`).

Los 9 métodos de la API están en la CLI:

| Método | CLI | Entrada |
|---|---|---|
| `inscribir(id, foto)` / `verificar(id, foto)` / `identificar(foto)` | `inscribir ID FOTO`, ... | foto completa |
| `inscribir_sd` / `verificar_sd` / `identificar_sd` | `verificar_sd ID CARA` | recorte de cara alineado (sin detector) |
| `inscribir_t` / `verificar_t` / `identificar_t` | `verificar_t ID TEMPLATE` | template `.npy`, `.csv` o `.txt` |

Utilidades para preparar las entradas de los métodos `_sd` y `_t`:

```bash
bb recortar /pruebas/vini.jpg /pruebas/cara.png       # recorte 128x128 que esperan los *_sd
bb extraer  /pruebas/vini.jpg /pruebas/t.npy          # template MLBP que esperan los *_t
bb extraer  --sd /pruebas/cara.png /pruebas/t.csv     # template desde un recorte
bb verificar_t vinicius_junior /pruebas/t.npy
bb inscribir vinicius_junior /pruebas/otra.jpg --actualizar   # añadir muestra (debe verificar)
bb construir_bd --force                               # regenerar BD (y clave)
docker run --rm --entrypoint pytest bigbrother -q     # tests (incluye ataques a la BD)
```

Salida: una línea JSON. Código de salida: `0` aceptado/ok, `1` rechazado/desconocido,
`2` error (sin cara, varias caras, cara de baja calidad, entrada inválida, BD manipulada,
inscripción rechazada…).

## 2. Ejecutar en local (Python 3.12+)

```bash
pip install -r requirements.txt
python scripts/build_db.py            # crea data/db.csv y keys/ (clave nueva)
python -m bigbrother verificar vinicius_junior data/fotos/vinicius_junior/vini_06.jpg
pytest -q
```

Desde Python:

```python
from bigbrother import BigBrother
bb = BigBrother()                      # carga data/db.csv con la clave de keys/
bb.verificar("vinicius_junior", "foto.jpg")   # -> bool  (acepta ruta o array BGR)
bb.identificar("foto.jpg")                     # -> id o None
cara = bb.recortar("foto.jpg");  bb.verificar_sd("vinicius_junior", cara)
t = bb.extraer("foto.jpg");      bb.verificar_t("vinicius_junior", t)
```

Excepciones (todas heredan de `BigBrotherError`): `NoFaceDetected`, `AmbiguousFace`,
`LowQualityFace`, `InvalidInput`, `EnrollmentRejected`, `IntegrityError`.

## 3. Reglas del sistema (para que no os sorprendan)

- La API **no devuelve puntuaciones**, solo la decisión (`bool` / id / `None`).
- Umbrales de similitud coseno (`models/calibration.json`): verificación con **FAR = 0,1 %** en
  LFW (identidades no vistas en el entrenamiento); identificación más estricta (FPIR 1 % contra toda la galería).
- Recortes (`_sd`) y templates (`_t`) que no parecen una cara → `NoFaceDetected`.
- `inscribir` sobre un id existente → rechazado. Con `actualizar=True` solo se acepta si la
  nueva foto **ya verifica** contra ese id.
- Inscribir a alguien que ya coincide con otra identidad → rechazado (deduplicación).
- Ids: `^[a-z0-9][a-z0-9_.-]{0,63}$`.
- Fotos con **dos caras de tamaño parecido**, caras de < 40 px o muy giradas → error.
- Los templates de `_t` deben ser histogramas MLBP válidos (forma, suman 1 por región,
  múltiplos de 1/256): si no, `InvalidInput`.
- La BD (`data/db.csv`) está cifrada y su huella se guarda en `keys/` (zona protegida). Si
  alguien la modifica, borra, restaura o copia, el sistema **se niega a arrancar**.

## 4. Reproducir entrenamiento y evaluación

```bash
pip install -r requirements-dev.txt
python -c "from sklearn.datasets import fetch_lfw_people as f; f(data_home='~/.cache/bigbrother', funneled=False)"
python scripts/prepare_lfw.py          # detecta y alinea las 13.233 caras de LFW
python scripts/train_projection.py --baseline --save    # -> models/mlbp_projection.npz, calibration.json
python scripts/train_faceness.py       # -> models/faceness.npz
python scripts/calibrate_identification.py
python scripts/evaluate.py             # -> docs/resultados/ (métricas, ataques, gráficas)
```

## Estructura

```
bigbrother/   system.py (API), detector.py, preprocess.py, mlbp.py, matcher.py,
              security.py (cifrado + integridad), database.py (CSV), __main__.py (CLI)
scripts/      build_db.py, prepare_lfw.py, train_projection.py, train_faceness.py,
              calibrate_identification.py, evaluate.py
models/       YuNet (.onnx), proyección MLBP→LDA, filtro de cara, calibración de umbrales
data/fotos/   dataset/ (49 fotos del docente) + vinicius_junior/ (5 fotos nuestras)
docs/         INFORME.md (vulnerabilidades/fortalezas) + resultados/
```
