"""glibc `rand()` (TYPE_3 additive feedback generator), as Tissue calls it unseeded."""

import numba
import numpy as np
from numpy.typing import NDArray

_DEGREE = 31
_SEPARATION = 3
_DISCARD = 310


def make_rand_state(seed: int = 1) -> NDArray[np.int64]:
    """Returns the generator state after glibc's `srand(seed)`.

    The first 31 entries are the ring buffer; the last is the step counter.
    """
    r = np.zeros(_DEGREE + 1, dtype=np.int64)
    r[0] = seed if seed != 0 else 1
    for i in range(1, _DEGREE):
        # 16807 * r[i-1] % (2^31 - 1), evaluated as glibc does (Schrage's method).
        hi, lo = divmod(int(r[i - 1]), 127773)
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        r[i] = word
    # Words 31-33 copy words 0-2, which already sit in those ring slots.
    r[_DEGREE] = _DEGREE + _SEPARATION
    for _ in range(_DISCARD):
        _next(r)
    return r


@numba.njit(cache=True)
def _next(r: NDArray[np.int64]) -> int:
    """Advances the state one step and returns the raw 32-bit word."""
    i = r[_DEGREE]
    word = (r[i % _DEGREE] + r[(i - _SEPARATION) % _DEGREE]) & 0xFFFFFFFF
    r[i % _DEGREE] = word
    r[_DEGREE] = i + 1
    return word


@numba.njit(cache=True)
def glibc_rand(r: NDArray[np.int64]) -> int:
    """Returns the next `rand()` value in [0, 2^31)."""
    return _next(r) >> 1
