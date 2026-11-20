"""Comparadores.

El sistema usa similitud coseno en el espacio proyectado (sqrt -> PCA blanqueado -> LDA),
que es la variante MLBP+LDA de Chan et al. (2007). chi2 se conserva como baseline clasico
(Ahonen et al., 2006) para la evaluacion.
"""
from pathlib import Path

import numpy as np

from . import config


def chi2(a, b, eps=1e-10):
    """Distancia chi-cuadrado (menor = mas parecido)."""
    return float(((a - b) ** 2 / (a + b + eps)).sum())


class Projection:
    """x -> normalize(W^T (sqrt(x) - mu)). Entrenada en LFW (scripts/train_projection.py)."""

    def __init__(self, path=config.PROJECTION_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}: ejecuta scripts/train_projection.py")
        z = np.load(path, allow_pickle=False)
        self.mean = z["mean"].astype(np.float32)
        self.W = z["W"].astype(np.float32)
        self.meta = {k: z[k].item() if z[k].ndim == 0 else tuple(z[k].tolist())
                     for k in z.files if k not in ("mean", "W")}
        self.in_dim, self.out_dim = self.W.shape

    def __call__(self, templates):
        x = np.atleast_2d(np.asarray(templates, np.float32))
        y = (np.sqrt(x) - self.mean) @ self.W
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
        return y if np.ndim(templates) == 2 else y[0]


class Faceness:
    """Clasificador lineal cara/no-cara sobre sqrt(template) (scripts/train_faceness.py)."""

    def __init__(self, path=config.FACENESS_PATH):
        z = np.load(path, allow_pickle=False)
        self.w, self.b, self.threshold = z["w"], float(z["b"]), float(z["threshold"])

    def score(self, template):
        return float(np.sqrt(np.asarray(template, np.float32)) @ self.w + self.b)


def cosine(a, b):
    return float(np.dot(a, b))
