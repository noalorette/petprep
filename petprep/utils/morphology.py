# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Simple morphological operations for binary masks."""

from __future__ import annotations

import numpy as np


def _clip(mask: np.ndarray, between: tuple = (0, 1)) -> np.ndarray:
    """Clip a mask to the given range."""
    return np.clip(mask, *between)


def dialate(mask: np.ndarray, by_radius: int) -> np.ndarray:
    """Dilate a binary mask by ``by_radius`` voxels."""
    from skimage.morphology import binary_dilation, ball

    out = binary_dilation(mask, ball(by_radius))
    return _clip(out)


def erode(mask: np.ndarray, by_radius: int) -> np.ndarray:
    """Erode a binary mask by ``by_radius`` voxels."""
    from skimage.morphology import binary_erosion, ball

    out = binary_erosion(mask, ball(by_radius))
    return _clip(out)