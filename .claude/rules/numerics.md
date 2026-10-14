# Numerics

## Fidelity comes first
Kernels that reproduce Tissue must keep its evaluation order: the same loop order, the same accumulation order and `libm` `pow`. A rewrite that changes any of these changes the last bit and breaks the bit-exact tests.
- Hot loops → Numba `@njit(cache=True)`, without `fastmath`.
- Sums → explicit `a*a + b*b + c*c`, or `np.bincount` weights, which add sequentially. Not `np.linalg.norm`, `einsum`, `np.add.reduceat` or `sum(axis=…)`, whose order differs.
- Powers in Hill terms → the Numba kernel (`feedback._hill_fraction`), since numpy's `power` differs from `libm` in the last bit.
- Run the slow fidelity tests (`uv run pytest -m slow`) after touching a kernel.
- Helpers called per cell or per triangle take scalars and tuples (`Vector`, `Matrix2`, `TriangleGeometry`). Numba counts references on every array argument per call: `compute_moduli(cells, c, …)` with the 28-array `CellState` cost 40 % of the force kernel. Small index lookups taking the mesh (`compute_triangle_geometry`, `get_rest_lengths`) are inlined by LLVM and measured free; anything larger, measure. The driver loops own the arrays and write the outputs.

## Elsewhere, leverage numpy
Outside the fidelity path (setup, IO, tests, analysis), prefer a numpy one-liner to a Python loop: `np.where`, broadcasting, fancy indexing, `np.bincount`, `np.linalg`. Mind memory on large meshes.
