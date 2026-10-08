import numpy as np


def chi2(a, b, weights=None, eps=1e-10):
    """Distancia chi-cuadrado (menor = mas parecido)."""
    d = (a - b) ** 2 / (a + b + eps)
    if weights is not None:
        d = d * weights
    return float(d.sum())
