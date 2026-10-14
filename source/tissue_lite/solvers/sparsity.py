"""Structure of the force Jacobian: its sparsity pattern and a column colouring
for finite differences."""

from collections.abc import Callable

import numba
import numpy as np
import scipy.sparse as sp

from tissue_lite.arrays import BoolArray, FloatArray, IntArray
from tissue_lite.model import ModelState


def build_jacobian_pattern(state: ModelState) -> sp.csc_array:
    """Returns the structural nonzeros of d(derivatives)/d(positions), both flattened.

    A node's force depends on the nodes it shares a centre triangle with: the
    centre and the two vertices. Tissue's sliver branch (KNOWN_ISSUES T3) couples
    further nodes and is not represented.
    """
    mesh = state.mesh
    num_vertices = state.tissue.topology.num_vertices
    num_cells = len(mesh.offsets) - 1
    centre = num_vertices + np.repeat(np.arange(num_cells), np.diff(mesh.offsets))
    nodes = (centre, mesh.vertex_a, mesh.vertex_b)
    rows = np.concatenate([nodes[i] for i in range(3) for _ in range(3)])
    cols = np.concatenate([nodes[j] for _ in range(3) for j in range(3)])
    n_nodes = num_vertices + num_cells
    node_pattern = sp.coo_array(
        (np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)
    )
    return sp.csc_array(sp.kron(node_pattern, np.ones((3, 3))))


def restrict_pattern(pattern: sp.csc_array, free: BoolArray) -> sp.csc_array:
    """Returns the pattern on the free rows and columns only."""
    return sp.csc_array(pattern[free][:, free])


def colour_columns(pattern: sp.csc_array) -> IntArray:
    """Greedy distance-2 colouring: columns sharing a row get different colours."""
    conflicts = sp.csr_array(pattern.T @ pattern)
    return _colour_greedily(conflicts.indptr, conflicts.indices, pattern.shape[1])


@numba.njit(cache=True)
def _colour_greedily(indptr: IntArray, indices: IntArray, n: int) -> IntArray:
    """Assigns each column the smallest colour unused by its conflicting columns."""
    colour = np.full(n, -1, dtype=np.int64)
    marked = np.full(n, -1, dtype=np.int64)  # colour -> last column marking it
    for j in range(n):
        for p in range(indptr[j], indptr[j + 1]):
            k = indices[p]
            if colour[k] >= 0:
                marked[colour[k]] = j
        c = 0
        while marked[c] == j:
            c += 1
        colour[j] = c
    return colour


def compute_jacobian(
    residual: Callable[[FloatArray], FloatArray],
    y: FloatArray,
    f: FloatArray,
    pattern: sp.csc_array,
    colours: IntArray,
) -> sp.csc_array:
    """Forward-difference Jacobian of `residual` at y (where it equals f).

    One evaluation per colour; the step per column is √ε scaled by max(|y|, 1).
    """
    eps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y), 1.0)
    diffs = np.empty((colours.max() + 1, len(y)))
    for k in range(diffs.shape[0]):
        diffs[k] = residual(y + np.where(colours == k, eps, 0.0)) - f
    coo = pattern.tocoo()
    data = diffs[colours[coo.col], coo.row] / eps[coo.col]
    return sp.csc_array((data, (coo.row, coo.col)), shape=pattern.shape)
