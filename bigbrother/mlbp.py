import numpy as np
from skimage.feature import local_binary_pattern

from . import config


def n_bins(points, method):
    if method == "nri_uniform":
        return points * (points - 1) + 3
    if method == "uniform":
        return points + 2
    if method == "default":
        return 2 ** points
    raise ValueError(method)


class MLBPExtractor:
    """Multiscale LBP (Chan et al., 2007): para cada radio se calcula el LBP, se divide la
    cara en una rejilla y se concatena el histograma normalizado de cada region.

    El "template" del sistema es este vector: len(radii) * filas * cols * bins floats,
    donde cada histograma regional suma 1.
    """

    def __init__(self, radii=config.LBP_RADII, points=config.LBP_POINTS, grid=config.GRID,
                 method=config.LBP_METHOD, size=config.FACE_SIZE):
        self.radii, self.points, self.grid, self.method, self.size = tuple(radii), points, grid, method, size
        self.bins = n_bins(points, method)
        self.n_cells = grid[0] * grid[1]
        self.n_hist = len(self.radii) * self.n_cells
        self.dim = self.n_hist * self.bins
        rows = (np.arange(size) * grid[0]) // size
        cols = (np.arange(size) * grid[1]) // size
        self._cell = (rows[:, None] * grid[1] + cols[None, :]).astype(np.intp)

    def extract(self, gray):
        """gray: uint8 (size x size) ya normalizado. Devuelve float32 (dim,)."""
        if gray.shape != (self.size, self.size):
            raise ValueError(f"Se esperaba una cara {self.size}x{self.size}, llego {gray.shape}")
        out = np.empty((len(self.radii), self.n_cells, self.bins), np.float32)
        for k, r in enumerate(self.radii):
            codes = local_binary_pattern(gray, self.points, r, method=self.method).astype(np.intp)
            idx = self._cell * self.bins + codes
            h = np.bincount(idx.ravel(), minlength=self.n_cells * self.bins)
            h = h.reshape(self.n_cells, self.bins).astype(np.float32)
            out[k] = h / h.sum(axis=1, keepdims=True)
        return out.ravel()

    def histograms(self, template):
        """Vista (n_hist, bins) de un template."""
        return np.asarray(template, np.float32).reshape(self.n_hist, self.bins)
