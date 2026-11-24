"""Alineado geometrico y normalizacion fotometrica."""
import cv2
import numpy as np

from . import config

# Posiciones objetivo de los 5 landmarks (fraccion del lado del recorte)
_TEMPLATE = np.array([[0.31, 0.38], [0.69, 0.38], [0.50, 0.56],
                      [0.35, 0.78], [0.65, 0.78]], dtype=np.float32)


def align(img_bgr, landmarks, size=config.FACE_SIZE):
    """Recorte alineado (BGR, size x size) por similitud a partir de 5 landmarks.

    Si la cara de entrada supera la resolucion del recorte (escala < 1), se pre-reduce
    la region de interes con INTER_AREA antes de warpAffine para evitar aliasing de alta
    frecuencia en el extractor MLBP.
    """
    M, _ = cv2.estimateAffinePartial2D(landmarks, _TEMPLATE * size, method=cv2.LMEDS)
    if M is None:
        raise ValueError("No se pudo estimar la transformacion de alineado")
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    if scale < 1.0:
        M_inv = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]],
                           dtype=np.float64)
        src_pts = corners @ M_inv.T
        pad = int(np.ceil(4.0 / scale))
        h, w = img_bgr.shape[:2]
        x0 = max(0, int(np.floor(src_pts[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(src_pts[:, 1].min())) - pad)
        x1 = min(w, int(np.ceil(src_pts[:, 0].max())) + pad)
        y1 = min(h, int(np.ceil(src_pts[:, 1].max())) + pad)
        if x1 > x0 and y1 > y0:
            roi = img_bgr[y0:y1, x0:x1]
            M = M.copy()
            M[:, 2] += M[:, 0] * x0 + M[:, 1] * y0
            rh, rw = roi.shape[:2]
            nw, nh = max(1, round(rw * scale)), max(1, round(rh * scale))
            sx, sy = nw / rw, nh / rh
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
            M[:, 2] += M[:, 0] * (0.5 / sx - 0.5) + M[:, 1] * (0.5 / sy - 0.5)
            M[:, 0] /= sx
            M[:, 1] /= sy
            img_bgr = roi
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
