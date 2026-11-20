"""Carga y validacion de imagenes de entrada."""
from pathlib import Path

import cv2
import numpy as np

from . import config
from .errors import InvalidInput


def load_image(path):
    """Lee una imagen como array BGR uint8. Soporta JPG/PNG/WebP y, via Pillow, AVIF/otros.

    (En el dataset hay un AVIF con extension .jpg: cv2.imread no lo abre.)
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidInput(f"No existe la imagen {path}")
    if path.stat().st_size > config.MAX_FILE_BYTES:
        raise InvalidInput(f"Imagen demasiado grande ({path.stat().st_size} bytes)")
    data = np.fromfile(str(path), dtype=np.uint8)  # soporta rutas unicode en Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_SIDE ** 2  # anti bomba de descompresion
            with Image.open(path) as im:
                img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:  # noqa: BLE001
            raise InvalidInput(f"No se pudo leer la imagen {path}: {e}") from e
    return img


def as_bgr(img):
    """Valida y normaliza una imagen de entrada a BGR uint8 de tamano razonable."""
    if isinstance(img, (str, Path)):
        img = load_image(img)
    if not isinstance(img, np.ndarray):
        raise InvalidInput("La foto debe ser un numpy.ndarray (BGR) o una ruta")
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and np.isfinite(img).all() and img.max() <= 1.0:
            img = (img * 255).round()
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif not (img.ndim == 3 and img.shape[2] == 3):
        raise InvalidInput(f"Forma de imagen no soportada: {img.shape}")
    h, w = img.shape[:2]
    if min(h, w) < config.MIN_IMAGE_SIDE:
        raise InvalidInput(f"Imagen demasiado pequena ({w}x{h})")
    if max(h, w) > config.MAX_IMAGE_SIDE:
        s = config.MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)
