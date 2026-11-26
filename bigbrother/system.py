"""BigBrother: sistema de reconocimiento facial basado en Multiscale LBP.

Pipeline:  foto -> YuNet (deteccion + 5 landmarks + calidad) -> alineado 128x128
           -> ecualizacion de histograma -> MLBP (radios 1-6, rejilla 8x8, 59 bins) = TEMPLATE
           -> proyeccion sqrt/PCA-blanqueado/LDA (entrenada en LFW) -> coseno -> umbral

Contrato de la API (todas las fotos: ruta o array BGR de OpenCV):
    inscribir(id, foto, actualizar=False) -> None   (lanza EnrollmentRejected si la politica lo impide)
    verificar(id, foto)                   -> bool
    identificar(foto)                     -> id | None
    *_sd: igual pero la foto es YA un recorte de cara (usar recortar() para obtenerlo)
    *_t : igual pero recibe el template (vector MLBP, usar extraer()/extraer_sd())

Decisiones de seguridad (ver docs/INFORME.md):
    - La API publica NO devuelve puntuaciones (evita ataques hill-climbing).
    - Un id ya inscrito no se puede sobrescribir/ampliar salvo con actualizar=True y una
      muestra que YA verifique contra esa identidad.
    - Nuevas identidades se deduplican contra la BD.
    - Templates validados (forma, rango, histogramas que suman 1, reticula 1/256).
    - BD cifrada (AES-GCM, id como AAD) + huella HMAC en la zona protegida: fail-closed.
"""
from pathlib import Path

import numpy as np

from . import config
from .database import Database
from .detector import FaceDetector
from .errors import EnrollmentRejected, InvalidInput, NoFaceDetected
from .imageio import as_bgr
from .matcher import Faceness, Projection
from .mlbp import MLBPExtractor
from .preprocess import align, normalize_face
from .security import KeyStore, validate_id


class BigBrother:
    def __init__(self, db_path=None, keys_dir=None, threshold=None, ident_threshold=None,
                 dedup_threshold=None, detector=None, extractor=None, projection=None):
        db_path = Path(db_path or config.DB_PATH)
        # Solo se genera una clave nueva si aun no existe la BD (si no, seria una clave
        # que no puede descifrarla).
        self.keystore = KeyStore(keys_dir or config.KEYS_DIR, create=not db_path.exists())
        # La BD comprueba tambien que modelos y umbrales (models/) no han cambiado desde que
        # se creo: editar calibration.json o borrar faceness.npz -> IntegrityError.
        self.db = Database(db_path, self.keystore)
        self.threshold = config.THRESHOLD if threshold is None else threshold
        self.ident_threshold = config.IDENT_THRESHOLD if ident_threshold is None else ident_threshold
        self.dedup_threshold = config.DEDUP_THRESHOLD if dedup_threshold is None else dedup_threshold
        self._detector = detector
        self.projection = projection or Projection()
        self.faceness = Faceness()  # obligatorio: sin el, *_sd/*_t aceptarian basura
        self.extractor = extractor or MLBPExtractor()
        meta = self.projection.meta
        if meta and (tuple(meta.get("radii", ())) != self.extractor.radii
                     or meta.get("method") != self.extractor.method):
            raise ValueError("La proyeccion no corresponde a la configuracion del extractor")
        if self.projection.in_dim != self.extractor.dim:
            raise ValueError("Dimension de la proyeccion incompatible con el extractor")

    @property
    def detector(self):
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    # ------------------------------------------------------------------ utilidades
    def recortar(self, foto, strict=True):
        """Detecta, controla calidad y alinea. Devuelve el recorte BGR (128x128) que
        esperan los metodos *_sd.

        Cuando YuNet devuelve varias detecciones, descarta con faceness los falsos positivos
        (p. ej. objetos grandes del fondo) antes de seleccionar la cara principal.
        """
        img = as_bgr(foto)
        faces = self.detector.detect_all(img)
        if len(faces) > 1:
            crops = {}

            def _is_face(face):
                crop = align(img, face.landmarks)
                crops[id(face)] = crop
                return self.faceness.score(self.extraer_sd(crop)) >= self.faceness.threshold

            face = self.detector.select(faces, strict=strict, is_face=_is_face)
            return crops[id(face)]
        face = self.detector.select(faces, strict=strict)
        return align(img, face.landmarks)

    def extraer_sd(self, cara):
        """Recorte de cara -> template MLBP (float32, %d floats)."""
        return self.extractor.extract(normalize_face(as_bgr(cara)))

    def extraer(self, foto):
        """Foto completa -> template MLBP."""
        return self.extraer_sd(self.recortar(foto))

    def identidades(self):
        return self.db.ids()

    def validar_template(self, template):
        """Un template valido es exactamente lo que produce el extractor: dim correcta,
        finito, no negativo, cada histograma regional suma 1 y sus valores son multiplos de
        1/n_pixeles_region. Esto impide inyectar vectores optimizados arbitrarios."""
        try:
            t = np.asarray(template, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise InvalidInput("Template no numerico") from e
        if t.shape != (self.extractor.dim,):
            raise InvalidInput(f"Template con forma {t.shape}, se esperaba ({self.extractor.dim},)")
        if not np.isfinite(t).all() or (t < 0).any():
            raise InvalidInput("Template con valores no finitos o negativos")
        H = t.reshape(self.extractor.n_hist, self.extractor.bins)
        if not np.allclose(H.sum(1), 1.0, atol=1e-4):
            raise InvalidInput("Cada histograma regional del template debe sumar 1")
        px = (config.FACE_SIZE // config.GRID[0]) * (config.FACE_SIZE // config.GRID[1])
        if not np.allclose(H * px, np.round(H * px), atol=1e-3):
            raise InvalidInput("El template no es un histograma MLBP valido")
        return t.astype(np.float32)

    # ------------------------------------------------------------------ puntuaciones
    # Uso interno/evaluacion. No forman parte de la API publica (no exponer scores).
    def _proj(self, template):
        t = self.validar_template(template)
        if self.faceness.score(t) < self.faceness.threshold:
            # Los metodos *_sd / *_t se saltan el detector: sin este filtro, ruido aleatorio
            # llegaba a identificarse como una persona inscrita.
            raise NoFaceDetected("El recorte/template no parece una cara")
        return self.projection(t)

    def _score_id(self, id_, y):
        refs = self.db.get(id_)
        return max((float(np.dot(y, r)) for r in refs), default=-1.0)

    def _scores(self, y):
        best = {}
        for id_, r in self.db.all():
            s = float(np.dot(y, r))
            if s > best.get(id_, -2.0):
                best[id_] = s
        return best

    def _identify_scored(self, template):
        scores = self._scores(self._proj(template))
        if not scores:
            return None, -1.0
        best_id = max(scores, key=scores.get)
        return best_id, scores[best_id]

    # ------------------------------------------------------------------ nivel template
    def inscribir_t(self, id, template, actualizar=False):
        validate_id(id)
        y = self._proj(template)
        if self.db.get(id):
            if not actualizar:
                raise EnrollmentRejected(f"'{id}' ya esta inscrito (usa actualizar=True)")
            if self._score_id(id, y) < self.threshold:
                raise EnrollmentRejected("La nueva muestra no coincide con la identidad inscrita")
        elif config.DEDUP_ENABLED and self.db.all():
            s = self._scores(y)
            if max(s.values()) >= self.dedup_threshold:
                raise EnrollmentRejected("La cara ya coincide con una identidad inscrita (duplicado)")
        self.db.add(id, y)

    def verificar_t(self, id, template):
        validate_id(id)
        y = self._proj(template)
        return self._score_id(id, y) >= self.threshold

    def identificar_t(self, template):
        best_id, s = self._identify_scored(template)
        return best_id if s >= self.ident_threshold else None

    # ------------------------------------------------------------------ sin detector
    def inscribir_sd(self, id, foto, actualizar=False):
        self.inscribir_t(id, self.extraer_sd(foto), actualizar)

    def verificar_sd(self, id, foto):
        return self.verificar_t(id, self.extraer_sd(foto))

    def identificar_sd(self, foto):
        return self.identificar_t(self.extraer_sd(foto))

    # ------------------------------------------------------------------ pipeline completo
    def inscribir(self, id, foto, actualizar=False):
        validate_id(id)
        self.inscribir_t(id, self.extraer(foto), actualizar)

    def verificar(self, id, foto):
        validate_id(id)
        return self.verificar_t(id, self.extraer(foto))

    def identificar(self, foto):
        return self.identificar_t(self.extraer(foto))

    # ------------------------------------------------------------------ administracion
    def _inscribir_admin(self, items):
        """Carga inicial (scripts/build_db.py): [(id, template)] sin politica de inscripcion."""
        self.db.add_many([(validate_id(i), self._proj(t)) for i, t in items])
