from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "db.csv"
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

FACE_SIZE = 128            # lado de la cara alineada (px)
LBP_POINTS = 8             # vecinos por patron LBP
LBP_RADII = (1, 2, 3)      # escalas (multiscale)
GRID = (8, 8)              # regiones (filas, columnas)
DETECTOR_SCORE_MIN = 0.7
THRESHOLD = 0.0            # TODO: calibrar con la evaluacion (distancia chi2)
