from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Face:
    bbox: tuple          # x, y, w, h
    landmarks: np.ndarray  # (5, 2): ojo dcho, ojo izq, nariz, boca dcha, boca izq
    score: float


class FaceDetector:
    """Detector de caras basado en YuNet (cv2.FaceDetectorYN)."""

    def __init__(self, model_path=config.YUNET_PATH, score_min=config.DETECTOR_SCORE_MIN):
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_min, 0.3, 5000)

    def detect(self, img_bgr):
        """Devuelve la cara con mayor score, o None."""
        h, w = img_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[-1])
        return Face(tuple(f[:4]), f[4:14].reshape(5, 2).astype(np.float32), float(f[-1]))
