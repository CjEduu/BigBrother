"""Excepciones del sistema. Todas heredan de BigBrotherError."""


class BigBrotherError(Exception):
    pass


class InvalidInput(BigBrotherError):
    """Entrada mal formada (imagen, id o template)."""


class NoFaceDetected(BigBrotherError):
    """El detector no encontro ninguna cara valida."""


class AmbiguousFace(BigBrotherError):
    """Hay varias caras de tamano similar: no se sabe a quien verificar."""


class LowQualityFace(BigBrotherError):
    """Cara demasiado pequena o de perfil (failure to acquire)."""


class EnrollmentRejected(BigBrotherError):
    """Inscripcion rechazada por la politica (id existente, duplicado, ...)."""


class IntegrityError(BigBrotherError):
    """La BD ha sido modificada fuera del sistema o no se puede descifrar."""
