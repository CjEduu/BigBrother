import numpy as np

from . import config
from .database import Database
from .detector import FaceDetector
from .matcher import chi2
from .mlbp import MLBPExtractor
from .preprocess import align, prepare_cropped


class NoFaceDetected(Exception):
    pass


class BigBrother:
    """Sistema de reconocimiento facial MLBP.

    Contrato: verificar* -> (bool, distancia); identificar* -> (id | None, distancia).
    Las fotos son arrays BGR de OpenCV (cv2.imread).
    """

    def __init__(self, db_path=config.DB_PATH, threshold=config.THRESHOLD,
                 detector=None, extractor=None):
        self.db = Database(db_path)
        self.threshold = threshold
        self._detector = detector
        self.extractor = extractor or MLBPExtractor()

    @property
    def detector(self):
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    # --- pipeline ---
    def _template_foto(self, foto):
        face = self.detector.detect(foto)
        if face is None:
            raise NoFaceDetected("No se detecto ninguna cara")
        return self.extractor.extract(align(foto, face.landmarks))

    def _template_sd(self, foto):
        return self.extractor.extract(prepare_cropped(foto))

    # --- nivel template (sin detector ni extractor) ---
    def inscribir_t(self, id, template):
        self.db.add(id, template)

    def verificar_t(self, id, template):
        refs = self.db.get(id)
        if not refs:
            return False, float("inf")
        d = min(chi2(template, r) for r in refs)
        return d <= self.threshold, d

    def identificar_t(self, template):
        best_id, best = None, float("inf")
        for id_, r in self.db.all():
            d = chi2(template, r)
            if d < best:
                best_id, best = id_, d
        return (best_id if best <= self.threshold else None), best

    # --- sin detector ---
    def inscribir_sd(self, id, foto):
        self.inscribir_t(id, self._template_sd(foto))

    def verificar_sd(self, id, foto):
        return self.verificar_t(id, self._template_sd(foto))

    def identificar_sd(self, foto):
        return self.identificar_t(self._template_sd(foto))

    # --- pipeline completo ---
    def inscribir(self, id, foto):
        self.inscribir_t(id, self._template_foto(foto))

    def verificar(self, id, foto):
        return self.verificar_t(id, self._template_foto(foto))

    def identificar(self, foto):
        return self.identificar_t(self._template_foto(foto))
