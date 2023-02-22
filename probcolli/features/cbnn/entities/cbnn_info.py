import numpy as np

from typing import NamedTuple


class CBNNInfo(NamedTuple):
    decision: np.ndarray
    mean: np.ndarray
    deviation: np.ndarray
    variance: np.ndarray
