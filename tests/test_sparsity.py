"""Tests for the force Jacobian pattern, colouring and compressed differences."""

import numpy as np
from conftest import CASES_DIR, case_state

from tissue_lite.solvers.sparsity import (
    build_jacobian_pattern,
    colour_columns,
    compute_jacobian,
)


def test_pattern_couples_the_nodes_of_every_triangle() -> None:
    state = case_state(CASES_DIR / "sweep_pre")
    pattern = build_jacobian_pattern(state).tocoo()
    num_vertices = state.tissue.topology.num_vertices
    mesh = state.mesh
    coupled = set(zip(pattern.row // 3, pattern.col // 3, strict=True))
    for c in range(len(mesh.offsets) - 1):
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            nodes = (num_vertices + c, mesh.vertex_a[t], mesh.vertex_b[t])
            assert all((i, j) in coupled for i in nodes for j in nodes)
    assert (pattern != pattern.T).nnz == 0


def test_colouring_separates_columns_sharing_a_row() -> None:
    pattern = build_jacobian_pattern(case_state(CASES_DIR / "sweep_pre"))
    colours = colour_columns(pattern)
    for i in range(pattern.shape[0]):
        columns = pattern[[i], :].tocoo().col
        assert len(set(colours[columns])) == len(columns)


def test_compressed_jacobian_matches_dense_differences() -> None:
    state = case_state(CASES_DIR / "sweep_pre")
    state.pressure_factor = 1.0
    y = state.positions.ravel() * 1.02
    trial = np.empty_like(state.positions)

    def residual(x: np.ndarray) -> np.ndarray:
        trial.ravel()[:] = x
        return state.compute_trial_forces(trial).ravel()

    f = residual(y)
    pattern = build_jacobian_pattern(state)
    sparse = compute_jacobian(residual, y, f, pattern, colour_columns(pattern))
    eps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y), 1.0)
    dense = np.column_stack(
        [
            (residual(y + eps[j] * np.eye(1, len(y), j)[0]) - f) / eps[j]
            for j in range(len(y))
        ]
    )
    scale = np.abs(dense).max()
    np.testing.assert_allclose(sparse.toarray(), dense, atol=1e-5 * scale)
