import numpy as np
from skimage.feature import local_binary_pattern

from . import config


class MLBPExtractor:
    """Multiscale LBP: histogramas LBP uniformes por region y por escala, concatenados."""

    def __init__(self, radii=config.LBP_RADII, points=config.LBP_POINTS, grid=config.GRID):
        self.radii, self.points, self.grid = radii, points, grid
        self.bins = points + 2
        self.dim = len(radii) * grid[0] * grid[1] * self.bins

    def extract(self, gray):
        """gray: uint8 2D (FACE_SIZE x FACE_SIZE). Devuelve vector float32 de longitud self.dim."""
        gh, gw = self.grid
        h, w = gray.shape
        parts = []
        for r in self.radii:
            lbp = local_binary_pattern(gray, self.points, r, method="uniform").astype(np.int64)
            for i in range(gh):
                for j in range(gw):
                    cell = lbp[i * h // gh:(i + 1) * h // gh, j * w // gw:(j + 1) * w // gw]
                    hist = np.bincount(cell.ravel(), minlength=self.bins).astype(np.float32)
                    parts.append(hist / max(hist.sum(), 1.0))
        return np.concatenate(parts).astype(np.float32)
