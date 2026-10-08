import cv2
import numpy as np

from . import config

# Posiciones objetivo de los 5 landmarks (fraccion del lado)
_TEMPLATE = np.array([[0.31, 0.38], [0.69, 0.38], [0.50, 0.56],
                      [0.35, 0.78], [0.65, 0.78]], dtype=np.float32)


def _gray_eq(img, size):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img)


def align(img_bgr, landmarks, size=config.FACE_SIZE):
    """Alinea la cara con una transformacion de similitud a partir de 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(landmarks, _TEMPLATE * size, method=cv2.LMEDS)
    warped = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return _gray_eq(warped, size)


def prepare_cropped(img, size=config.FACE_SIZE):
    """Para los metodos *_sd: la imagen ya es una cara recortada."""
    return _gray_eq(img, size)
