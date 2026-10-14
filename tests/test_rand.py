"""Tests for the glibc `rand()` port."""

import numpy as np

from tissue_lite.rand import glibc_rand, make_rand_state

# glibc 2.39 rand() after srand(1).
GLIBC_SEED_1 = [1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335]


def test_matches_glibc_sequence() -> None:
    state = make_rand_state()
    assert [glibc_rand(state) for _ in GLIBC_SEED_1] == GLIBC_SEED_1


def test_states_are_independent() -> None:
    a, b = make_rand_state(), make_rand_state()
    glibc_rand(a)
    assert not np.array_equal(a, b)
    assert glibc_rand(b) == GLIBC_SEED_1[0]


def test_values_are_in_range() -> None:
    state = make_rand_state(seed=7)
    values = [glibc_rand(state) for _ in range(1000)]
    assert min(values) >= 0
    assert max(values) <= 2**31 - 1
