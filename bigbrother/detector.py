from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .errors import AmbiguousFace, LowQualityFace, NoFaceDetected

try:  # OpenCV 5 avisa por stderr de "Targets are not supported..." en cada creacion
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


@dataclass
class Face:
    bbox: tuple            # x, y, w, h
    landmarks: np.ndarray  # (5, 2): ojo dcho, ojo izq, nariz, boca dcha, boca izq
    score: float

    @property
    def area(self):
        return float(self.bbox[2] * self.bbox[3])


class FaceDetector:
    """Detector YuNet (cv2.FaceDetectorYN) + controles de calidad."""

    def __init__(self, model_path=config.YUNET_PATH, score_min=config.DETECTOR_SCORE_MIN):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Falta el modelo {model_path}. Ejecuta: python scripts/download_models.py")
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_min, 0.3, 5000)

    def detect_all(self, img_bgr):
        h, w = img_bgr.shape[:2]
        s = min(1.0, config.DETECT_MAX_SIDE / max(h, w))
        small = img_bgr if s == 1.0 else cv2.resize(img_bgr, (round(w * s), round(h * s)),
                                                     interpolation=cv2.INTER_AREA)
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._det.detect(small)
        if faces is None:
            return []
        # coordenadas devueltas en la imagen original (el alineado usa la resolucion completa)
        out = [Face(tuple(float(v) / s for v in f[:4]), (f[4:14].reshape(5, 2) / s).astype(np.float32),
                    float(f[-1])) for f in faces]
        return sorted(out, key=lambda f: f.area, reverse=True)

    def detect(self, img_bgr, strict=True):
        """Devuelve la cara principal (la mayor).

        strict=True aplica la politica de seguridad: rechaza imagenes con varias caras de
        tamano parecido (ataque "pon una foto de la victima junto a tu cara") y caras de
        baja calidad (muy pequenas o de perfil).
        """
        faces = self.detect_all(img_bgr)
        if not faces:
            raise NoFaceDetected("No se detecto ninguna cara")
        face = faces[0]
        if strict:
            if len(faces) > 1 and faces[1].area >= config.AMBIGUOUS_FACE_RATIO * face.area:
                raise AmbiguousFace(f"{len(faces)} caras de tamano similar en la imagen")
            check_quality(face)
        return face


def check_quality(face):
    x, y, w, h = face.bbox
    if min(w, h) < config.MIN_FACE_PX:
        raise LowQualityFace(f"Cara demasiado pequena ({w:.0f}x{h:.0f}px)")
    re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_d = float(np.linalg.norm(le - re))
    if eye_d < config.MIN_EYE_DIST_PX:
        raise LowQualityFace(f"Distancia interocular insuficiente ({eye_d:.1f}px)")
    # yaw aproximado: desplazamiento de la nariz respecto al centro de los ojos,
    # medido sobre el eje de los ojos (invariante a la rotacion en el plano)
    axis = (le - re) / eye_d
    yaw = abs(float(np.dot(nose - (le + re) / 2, axis))) / eye_d
    if yaw > config.MAX_YAW_RATIO:
        raise LowQualityFace(f"Cara demasiado girada (yaw ratio {yaw:.2f})")
