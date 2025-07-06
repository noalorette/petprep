import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, ball

from petprep.utils.morphology import dialate, erode


def test_dialate():
    mask = np.zeros((5, 5, 5), dtype=int)
    mask[2, 2, 2] = 1
    result = dialate(mask, 1)
    expected = binary_dilation(mask, ball(1))
    assert np.array_equal(result, expected)


def test_erode():
    mask = np.ones((5, 5, 5), dtype=int)
    mask[0, :, :] = 0
    result = erode(mask, 1)
    expected = binary_erosion(mask, ball(1))
    assert np.array_equal(result, expected)