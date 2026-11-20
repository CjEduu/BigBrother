from .errors import (AmbiguousFace, BigBrotherError, EnrollmentRejected, IntegrityError,
                     InvalidInput, LowQualityFace, NoFaceDetected)

__all__ = ["BigBrother", "BigBrotherError", "NoFaceDetected", "AmbiguousFace", "LowQualityFace",
           "EnrollmentRejected", "IntegrityError", "InvalidInput"]


def __getattr__(name):  # import perezoso: los scripts pueden usar submodulos sin el modelo
    if name == "BigBrother":
        from .system import BigBrother
        return BigBrother
    raise AttributeError(name)
