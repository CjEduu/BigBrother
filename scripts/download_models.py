"""Descarga el modelo YuNet (OpenCV Zoo) en models/."""
import urllib.request
from pathlib import Path

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")
DEST = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if not DEST.exists():
        urllib.request.urlretrieve(URL, DEST)
    print(f"OK: {DEST} ({DEST.stat().st_size} bytes)")
