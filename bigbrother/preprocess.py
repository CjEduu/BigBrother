"""Alineado geometrico y normalizacion fotometrica."""
import cv2
import numpy as np

from . import config

# Posiciones objetivo de los 5 landmarks (fraccion del lado del recorte)
_TEMPLATE = np.array([[0.31, 0.38], [0.69, 0.38], [0.50, 0.56],
                      [0.35, 0.78], [0.65, 0.78]], dtype=np.float32)


def align(img_bgr, landmarks, size=config.FACE_SIZE):
    """Recorte alineado (BGR, size x size) por similitud a partir de 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(landmarks, _TEMPLATE * size, method=cv2.LMEDS)
    if M is None:
        raise ValueError("No se pudo estimar la transformacion de alineado")
    return cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def to_gray(img, size=config.FACE_SIZE):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != (size, size):
        interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
        img = cv2.resize(img, (size, size), interpolation=interp)
    return img


def tan_triggs(gray, gamma=0.2, sigma0=1.0, sigma1=2.0, alpha=0.1, tau=10.0):
    """Normalizacion de iluminacion de Tan & Triggs (2010): gamma + DoG + ecualizacion de
    contraste. Devuelve uint8 para el LBP."""
    x = np.power(gray.astype(np.float32) + 1.0, gamma)
    x = cv2.GaussianBlur(x, (0, 0), sigma0) - cv2.GaussianBlur(x, (0, 0), sigma1)
    x = x / (np.mean(np.abs(x) ** alpha) ** (1 / alpha) + 1e-8)
    x = x / (np.mean(np.minimum(tau, np.abs(x)) ** alpha) ** (1 / alpha) + 1e-8)
    x = tau * np.tanh(x / tau)
    return np.clip((x + tau) * (255.0 / (2 * tau)), 0, 255).astype(np.uint8)


def photometric(gray, method=config.PHOTOMETRIC):
    if method == "tantriggs":
        return tan_triggs(gray)
    if method == "equalize":
        return cv2.equalizeHist(gray)
    if method == "none":
        return gray
    raise ValueError(method)


def normalize_face(face_img, method=config.PHOTOMETRIC, size=config.FACE_SIZE):
    """Recorte (alineado) -> gris size x size normalizado, listo para el extractor."""
    return photometric(to_gray(face_img, size), method)
