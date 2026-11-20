"""Configuracion global de BigBrother.

Las rutas sensibles (clave maestra y estado de la BD) se pueden mover con variables
de entorno para separarlas fisicamente de la BD (ver docs/INFORME_SEGURIDAD.md).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Rutas -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("BIGBROTHER_DATA", ROOT / "data"))
DB_PATH = Path(os.environ.get("BIGBROTHER_DB", DATA_DIR / "db.csv"))
# "Zona protegida": clave maestra + huella de la BD. NO debe estar en el mismo
# volumen/permisos que la BD (en docker: /app/keys, la BD en /app/data).
KEYS_DIR = Path(os.environ.get("BIGBROTHER_KEYS", ROOT / "keys"))
KEY_PATH = KEYS_DIR / "master.key"
STATE_PATH = KEYS_DIR / "db.state"

MODELS_DIR = ROOT / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
PROJECTION_PATH = MODELS_DIR / "mlbp_projection.npz"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
FACENESS_PATH = MODELS_DIR / "faceness.npz"

# --- Detector / calidad ----------------------------------------------------
DETECTOR_SCORE_MIN = 0.75     # confianza minima de YuNet
MIN_FACE_PX = 40              # lado minimo del bbox de la cara (px)
MIN_EYE_DIST_PX = 16          # distancia interocular minima (px)
MAX_YAW_RATIO = 0.45          # |nariz - centro ojos| / dist. ojos (descarta perfiles)
AMBIGUOUS_FACE_RATIO = 0.45   # si la 2a cara tiene >= 45% del area de la 1a -> ambiguo
MAX_IMAGE_SIDE = 6000         # imagenes mayores se reescalan (anti-DoS)
MIN_IMAGE_SIDE = 32
MAX_FILE_BYTES = 30 * 1024 * 1024
DETECT_MAX_SIDE = 1600        # se detecta sobre una copia reducida (velocidad / anti-DoS)

# --- Extractor MLBP --------------------------------------------------------
FACE_SIZE = 128               # lado de la cara alineada (px)
LBP_POINTS = 8                # vecinos por patron
LBP_RADII = (1, 2, 3, 4, 5, 6)  # escalas (multiscale)
LBP_METHOD = "nri_uniform"    # 59 bins para P=8 (uniformes no invariantes a rotacion)
GRID = (8, 8)                 # regiones (filas, columnas)
PHOTOMETRIC = "equalize"      # "equalize" | "tantriggs" (TT dio peor EER en LFW: 19.7% vs 15.5%)

# --- Politica de inscripcion ----------------------------------------------
ID_REGEX = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
# Al inscribir una identidad nueva se rechaza si la cara ya coincide con otra
# identidad inscrita con una similitud >= DEDUP_THRESHOLD (deduplicacion).
DEDUP_ENABLED = True


def _load_calibration():
    """Umbrales calibrados por scripts/train_projection.py (similitud coseno)."""
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# Los umbrales NO se pueden cambiar por variables de entorno (seria un bypass trivial) y el
# fichero de calibracion esta cubierto por la huella de modelos guardada en keys/db.state.
_CAL = _load_calibration()
THRESHOLD = float(_CAL.get("threshold", 0.5))
DEDUP_THRESHOLD = float(_CAL.get("dedup_threshold", THRESHOLD))
# identificacion open-set: umbral mas estricto (FPIR 1% contra la galeria)
IDENT_THRESHOLD = float(_CAL.get("ident_threshold", THRESHOLD))

# Ficheros cuya integridad se ata a la BD (cambiarlos invalida la BD existente)
MODEL_FILES = (YUNET_PATH, PROJECTION_PATH, FACENESS_PATH, CALIBRATION_PATH)
