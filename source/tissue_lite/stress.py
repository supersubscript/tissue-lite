"""Per-cell stress and strain tensors, principal axes and energies.

Ports the `update` part of `VertexFromTRBScenterTriangulationMT` from Tissue
e74e457, including its Jacobi eigensolver. Each triangle's Almansi strain and
Cauchy stress are rotated into global coordinates and averaged over the cell by
area; the principal axes come from Tissue's Jacobi loops.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.material import (
    Lame,
    Material,
    compute_anisotropic_stress,
    compute_fibre_direction,
    compute_fibre_strain,
    compute_lame_parameters,
    compute_moduli,
)
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    Matrix2,
    TriangleGeometry,
    TriangleMesh,
    Vector,
    compute_deformation,
    compute_green_strain,
    compute_triangle_area,
    compute_triangle_geometry,
    cross,
    divide,
    dot,
    get_rest_lengths,
    multiply,
    norm,
    transpose,
)

_EIGEN_TOLERANCE = 0.000001
_C45 = 0.70710678118
_ZERO_STRESS = 0.000001
_OUT_OF_PLANE = 0.7
_NO_PERPENDICULAR = 0.0001

type Matrix3 = tuple[Vector, Vector, Vector]
_ZERO3: Matrix3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@numba.njit(cache=True)
def update_cell_stress(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
) -> None:
    """Writes the area-weighted stress and strain of every cell into `cells`.

    Sets the (C, 3, 3) stress and strain tensors (Cauchy and Almansi, global axes), the
    normal, both principal axes of stress and strain (ranked by |value|, KNOWN_ISSUES
    T5; Jacobi signs as in Tissue, T4), their anisotropies, the von Mises stress (T10),
    the total rest area (T8) and the energies, and normalises `cells.microtubule`, as
    Tissue's `update` does. With `material.fidelity`, the anisotropic energy also goes
    into `young_transverse` (Tissue's slot 20, T7). A vanishing fibre direction
    consumes `rand_state` (T1).

    Raises:
        ZeroDivisionError: If a triangle is degenerate.
    """
    for c in range(mesh.offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(
            cells.young_longitudinal[c], cells.concentration[c], material
        )
        lame = compute_lame_parameters(young_l, young_t, material)
        microtubule = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        strain_sum = _ZERO3
        stress_sum = _ZERO3
        normal_sum = (0.0, 0.0, 0.0)
        total_rest_area = 0.0
        total_area = 0.0
        energy_iso = 0.0
        energy_aniso = 0.0
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            # Geometry, deformation and fibre direction of the triangle.
            triangle = compute_triangle_geometry(mesh, centres, vertices, c, t)
            rest_area, _, gradient = compute_deformation(
                triangle.lengths, get_rest_lengths(mesh, t)
            )
            area = compute_triangle_area(triangle.lengths)
            fibre = compute_fibre_direction(
                gradient,
                triangle.x_axis,
                triangle.y_axis,
                microtubule,
                rand_state,
                material.fidelity,
            )
            strain = compute_green_strain(gradient)
            left_cauchy_green = multiply(gradient, transpose(gradient))
            # In-plane Almansi strain and Cauchy stress in global axes.
            strain_global = _rotate_to_global(
                triangle, _compute_almansi_strain(left_cauchy_green)
            )
            stress_global = _rotate_to_global(
                triangle,
                _compute_cauchy_stress(
                    gradient, left_cauchy_green, strain, fibre, lame, rest_area / area
                ),
            )
            # Area-weighted sums over the cell.
            strain_sum = _add_scaled(strain_sum, area, strain_global)
            stress_sum = _add_scaled(stress_sum, area, stress_global)
            normal_sum = (
                normal_sum[0] + area * triangle.normal[0],
                normal_sum[1] + area * triangle.normal[1],
                normal_sum[2] + area * triangle.normal[2],
            )
            total_rest_area = total_rest_area + rest_area
            total_area = total_area + area
            iso, aniso = _compute_energy_densities(strain, fibre, lame)
            energy_iso += iso * rest_area
            energy_aniso += aniso * rest_area
        # Cell means, normal, principal axes and scalars.
        strain_mean = _divide3(strain_sum, total_area)
        stress_mean = _divide3(stress_sum, total_area)
        normal = _compute_unit_normal(normal_sum, total_area)
        _write_matrix(cells.stress_tensor[c], stress_mean)
        _write_matrix(cells.strain_tensor[c], strain_mean)
        cells.normal[c, 0] = normal[0]
        cells.normal[c, 1] = normal[1]
        cells.normal[c, 2] = normal[2]
        first, second, s1, s2 = _find_principal_stress(stress_mean, normal)
        cells.von_mises_stress[c] = np.sqrt(s1 * s1 + s2 * s2 - s1 * s2)
        cells.stress_anisotropy[c] = 0.0 if abs(s1) < _ZERO_STRESS else 1 - abs(s2 / s1)
        for d in range(3):
            cells.stress_direction[c, d] = first[d]
            cells.stress_direction_second[c, d] = second[d]
        cells.stress_max[c] = s1
        cells.stress_second[c] = s2
        axis, perpendicular, e1, e2 = _find_principal_strain(strain_mean, normal)
        cells.strain_anisotropy[c] = 0.0 if abs(e1) < _ZERO_STRESS else 1 - abs(e2 / e1)
        cells.strain_direction[c] = axis
        cells.strain_perpendicular[c] = perpendicular
        cells.strain_max[c] = e1
        cells.strain_second[c] = e2
        length = norm(microtubule)
        if length > 0:
            for d in range(3):
                cells.microtubule[c, d] = cells.microtubule[c, d] / length
        cells.rest_area[c] = total_rest_area
        cells.energy_isotropic[c] = energy_iso / total_rest_area
        cells.energy_anisotropic[c] = energy_aniso / total_rest_area
        if material.fidelity:
            cells.young_transverse[c] = energy_aniso / total_rest_area


@numba.njit(cache=True)
def _compute_almansi_strain(left_cauchy_green: Matrix2) -> Matrix2:
    """Returns e = (I - B⁻¹) / 2 for B = F Fᵀ."""
    (b00, b01), (b10, b11) = left_cauchy_green
    det_b = b00 * b11 - b10 * b01
    return (
        (0.5 * (1 - (b11 / det_b)), 0.5 * b01 / det_b),
        (0.5 * b10 / det_b, 0.5 * (1 - (b00 / det_b))),
    )


@numba.njit(cache=True)
def _compute_cauchy_stress(
    gradient: Matrix2,
    left_cauchy_green: Matrix2,
    strain: Matrix2,
    fibre: tuple[float, float],
    lame: Lame,
    area_factor: float,
) -> Matrix2:
    """Cauchy stress σ = (A_rest / A) F S Fᵀ, with S the Saint Venant-Kirchhoff stress
    of the transverse material plus the fibre stress, the stress of the force model
    (Delingette's springs are SVK)."""
    (b00, b01), (b10, b11) = left_cauchy_green
    (bb00, bb01), (bb10, bb11) = multiply(left_cauchy_green, left_cauchy_green)
    mu_t = lame.mu_t
    # Isotropic part written through B: λ tr E · B + μ (B² - B).
    g = lame.lambda_t * (strain[0][0] + strain[1][1]) - mu_t
    sg00 = area_factor * (g * b00 + mu_t * bb00)
    sg01 = area_factor * (g * b01 + mu_t * bb01)
    sg10 = area_factor * (g * b10 + mu_t * bb10)
    sg11 = area_factor * (g * b11 + mu_t * bb11)
    fibre_stress = compute_anisotropic_stress(
        strain, fibre, lame.lambda_l - lame.lambda_t, lame.mu_l - lame.mu_t
    )
    (d00, d01), (d10, d11) = multiply(
        gradient, multiply(fibre_stress, transpose(gradient))
    )
    return (
        (sg00 + area_factor * d00, sg01 + area_factor * d01),
        (sg10 + area_factor * d10, sg11 + area_factor * d11),
    )


@numba.njit(cache=True)
def _compute_energy_densities(
    strain: Matrix2, fibre: tuple[float, float], lame: Lame
) -> tuple[float, float]:
    """Isotropic and fibre strain energy per rest area (Tissue's `StrainEnergyIso`,
    `StrainEnergyAniso`), from the invariants tr E, tr E² and aᵀE²a."""
    (e00, _), (_, e11) = strain
    squared = multiply(strain, strain)
    trace = e00 + e11
    trace_squared = squared[0][0] + squared[1][1]
    fibre_strain = compute_fibre_strain(strain, fibre)
    fibre_squared = compute_fibre_strain(squared, fibre)
    delta_lambda = lame.lambda_l - lame.lambda_t
    delta_mu = lame.mu_l - lame.mu_t
    return (
        (lame.lambda_t / 2) * trace * trace + lame.mu_t * trace_squared,
        (delta_lambda / 2) * fibre_strain * trace + delta_mu * fibre_squared,
    )


@numba.njit(cache=True)
def _rotate_to_global(triangle: TriangleGeometry, local: Matrix2) -> Matrix3:
    """Returns R L Rᵀ for the in-plane tensor L and the frame R = (x, y, normal).

    Sums as Tissue does: from 0.0 and over L's zero row and column, so that signed
    zeros match.
    """
    row0 = (triangle.x_axis[0], triangle.y_axis[0], triangle.normal[0])
    row1 = (triangle.x_axis[1], triangle.y_axis[1], triangle.normal[1])
    row2 = (triangle.x_axis[2], triangle.y_axis[2], triangle.normal[2])
    column0 = (local[0][0], local[1][0], 0.0)
    column1 = (local[0][1], local[1][1], 0.0)
    column2 = (0.0, 0.0, 0.0)
    rl0 = (
        _dot_from_zero(row0, column0),
        _dot_from_zero(row0, column1),
        _dot_from_zero(row0, column2),
    )
    rl1 = (
        _dot_from_zero(row1, column0),
        _dot_from_zero(row1, column1),
        _dot_from_zero(row1, column2),
    )
    rl2 = (
        _dot_from_zero(row2, column0),
        _dot_from_zero(row2, column1),
        _dot_from_zero(row2, column2),
    )
    return (
        (
            _dot_from_zero(rl0, row0),
            _dot_from_zero(rl0, row1),
            _dot_from_zero(rl0, row2),
        ),
        (
            _dot_from_zero(rl1, row0),
            _dot_from_zero(rl1, row1),
            _dot_from_zero(rl1, row2),
        ),
        (
            _dot_from_zero(rl2, row0),
            _dot_from_zero(rl2, row1),
            _dot_from_zero(rl2, row2),
        ),
    )


@numba.njit(cache=True)
def _dot_from_zero(a: Vector, b: Vector) -> float:
    """a · b accumulated from 0.0 as Tissue's loops do (keeps the sign of zero)."""
    return 0.0 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.njit(cache=True)
def _add_scaled(total: Matrix3, weight: float, tensor: Matrix3) -> Matrix3:
    """Returns total + weight * tensor, elementwise."""
    return (
        (
            total[0][0] + weight * tensor[0][0],
            total[0][1] + weight * tensor[0][1],
            total[0][2] + weight * tensor[0][2],
        ),
        (
            total[1][0] + weight * tensor[1][0],
            total[1][1] + weight * tensor[1][1],
            total[1][2] + weight * tensor[1][2],
        ),
        (
            total[2][0] + weight * tensor[2][0],
            total[2][1] + weight * tensor[2][1],
            total[2][2] + weight * tensor[2][2],
        ),
    )


@numba.njit(cache=True)
def _divide3(total: Matrix3, divisor: float) -> Matrix3:
    """Returns total / divisor, dividing each entry."""
    (t00, t01, t02), (t10, t11, t12), (t20, t21, t22) = total
    return (
        (t00 / divisor, t01 / divisor, t02 / divisor),
        (t10 / divisor, t11 / divisor, t12 / divisor),
        (t20 / divisor, t21 / divisor, t22 / divisor),
    )


@numba.njit(cache=True)
def _write_matrix(target: FloatArray, matrix: Matrix3) -> None:
    """Copies the 3×3 tuple into the array."""
    for r in range(3):
        for q in range(3):
            target[r, q] = matrix[r][q]


@numba.njit(cache=True)
def _column(matrix: Matrix3, q: int) -> Vector:
    """Returns column q of the matrix."""
    return (matrix[0][q], matrix[1][q], matrix[2][q])


@numba.njit(cache=True)
def _compute_unit_normal(total: Vector, total_area: float) -> Vector:
    """Returns the unit cell normal from the area-weighted sum (0 stays 0)."""
    mean = divide(total, total_area)
    length = norm(mean)
    return divide(mean, length) if length > 0 else mean


@numba.njit(cache=True)
def _find_principal_stress(
    stress: Matrix3, normal: Vector
) -> tuple[Vector, Vector, float, float]:
    """Diagonalises the cell stress; returns the two in-plane principal axes and
    their stresses, largest magnitude first."""
    diagonal, eigenvectors = _jacobi(stress, True)
    first, second, third = _rank_principal_axes(diagonal, True)
    s1 = diagonal[first][first]
    s2 = diagonal[second][second]
    if abs(dot(normal, _column(eigenvectors, first))) > _OUT_OF_PLANE:
        first, second = second, third
        s1, s2 = s2, diagonal[third][third]
    return _column(eigenvectors, first), _column(eigenvectors, second), s1, s2


@numba.njit(cache=True)
def _find_principal_strain(
    strain: Matrix3, normal: Vector
) -> tuple[Vector, Vector, float, float]:
    """Diagonalises the cell strain; returns the principal axis, the in-plane
    perpendicular (unnormalised, KNOWN_ISSUES T9) and the two strains."""
    diagonal, eigenvectors = _jacobi(strain, False)
    first, second, _ = _rank_principal_axes(diagonal, False)
    axis = _column(eigenvectors, first)
    perpendicular = cross(normal, axis)
    if abs(norm(perpendicular)) < _NO_PERPENDICULAR:
        perpendicular = axis
    return axis, perpendicular, diagonal[first][first], diagonal[second][second]


@numba.njit(cache=True)
def _rank_principal_axes(diagonal: Matrix3, cyclic: bool) -> tuple[int, int, int]:
    """Orders the axes of a diagonalised tensor by |value| (KNOWN_ISSUES T5).

    On a tie between the two smaller ones, the stress loop (`cyclic`) keeps cyclic
    order and the strain loop ascending order.
    """
    first = 0
    if abs(diagonal[1][1]) > abs(diagonal[first][first]):
        first = 1
    if abs(diagonal[2][2]) > abs(diagonal[first][first]):
        first = 2
    if first == 0:
        second, third = 1, 2
    elif first == 1:
        second, third = (2, 0) if cyclic else (0, 2)
    else:
        second, third = 0, 1
    if abs(diagonal[third][third]) > abs(diagonal[second][second]):
        second, third = third, second
    return first, second, third


@numba.njit(cache=True)
def _pick_pivot(tensor: Matrix3) -> tuple[float, int, int]:
    """Returns the largest off-diagonal magnitude below the diagonal and its indices.

    Returns:
        (pivot, big, small): the magnitude, its row `big` and column `small`;
        `big > small`, and the pair names the plane the next rotation acts in.
    """
    pivot, big, small = abs(tensor[1][0]), 1, 0
    if abs(tensor[2][0]) > pivot:
        pivot, big, small = abs(tensor[2][0]), 2, 0
    if abs(tensor[2][1]) > pivot:
        pivot, big, small = abs(tensor[2][1]), 2, 1
    return pivot, big, small


@numba.njit(cache=True)
def _mix_columns(low: float, high: float, co: float, si: float) -> tuple[float, float]:
    """Rotates one row's pair of entries in columns (small, big).

    `low` and `high` are the entries in columns `small` and `big`; `co` and `si`
    are the rotation's cosine and sine. Returns the new (low, high).
    """
    return low * co + high * si, low * -si + high * co


@numba.njit(cache=True)
def _mix_rows(low: float, high: float, co: float, si: float) -> tuple[float, float]:
    """Rotates one column's pair of entries in rows (small, big).

    `low` and `high` are the entries in rows `small` and `big`; `co` and `si` are
    the rotation's cosine and sine. Returns the new (low, high).
    """
    return co * low + si * high, -si * low + co * high


@numba.njit(cache=True)
def _rotate(
    tensor: Matrix3, eigenvectors: Matrix3, big: int, small: int
) -> tuple[Matrix3, Matrix3]:
    """Applies Tissue's Jacobi rotation in the (small, big) plane.

    `tensor` is the symmetric 3×3 matrix being diagonalised and `eigenvectors`
    the accumulated rotations (eigenvectors as columns); `big > small` are the
    plane's axis indices from `_pick_pivot`. `aij` and `vij` are the row i, column
    j entries of the two matrices. Returns both matrices after the rotation.

    The rotation sign is lost as in Tissue (KNOWN_ISSUES T4). Columns rotate first,
    then rows; skipping Tissue's exact products with 0 and 1 leaves the result
    bit-identical. Unrolled per plane on scalars, since this loop dominates the
    stress update.
    """
    if abs(tensor[big][big] - tensor[small][small]) < _EIGEN_TOLERANCE:
        si = _C45
        co = _C45
    else:
        tan = (2 * tensor[big][small]) / (tensor[small][small] - tensor[big][big])
        tan = 1 / np.sqrt(1 + tan * tan)
        si = _C45 * np.sqrt(1 - tan) if tan < 1 else _C45 * np.sqrt(tan - 1)
        co = _C45 * np.sqrt(1 + tan)
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = tensor
    (v00, v01, v02), (v10, v11, v12), (v20, v21, v22) = eigenvectors
    if big == 1:  # small == 0
        a00, a01 = _mix_columns(a00, a01, co, si)
        a10, a11 = _mix_columns(a10, a11, co, si)
        a20, a21 = _mix_columns(a20, a21, co, si)
        a00, a10 = _mix_rows(a00, a10, co, si)
        a01, a11 = _mix_rows(a01, a11, co, si)
        a02, a12 = _mix_rows(a02, a12, co, si)
        v00, v01 = _mix_columns(v00, v01, co, si)
        v10, v11 = _mix_columns(v10, v11, co, si)
        v20, v21 = _mix_columns(v20, v21, co, si)
    elif small == 0:  # big == 2
        a00, a02 = _mix_columns(a00, a02, co, si)
        a10, a12 = _mix_columns(a10, a12, co, si)
        a20, a22 = _mix_columns(a20, a22, co, si)
        a00, a20 = _mix_rows(a00, a20, co, si)
        a01, a21 = _mix_rows(a01, a21, co, si)
        a02, a22 = _mix_rows(a02, a22, co, si)
        v00, v02 = _mix_columns(v00, v02, co, si)
        v10, v12 = _mix_columns(v10, v12, co, si)
        v20, v22 = _mix_columns(v20, v22, co, si)
    else:  # small == 1, big == 2
        a01, a02 = _mix_columns(a01, a02, co, si)
        a11, a12 = _mix_columns(a11, a12, co, si)
        a21, a22 = _mix_columns(a21, a22, co, si)
        a10, a20 = _mix_rows(a10, a20, co, si)
        a11, a21 = _mix_rows(a11, a21, co, si)
        a12, a22 = _mix_rows(a12, a22, co, si)
        v01, v02 = _mix_columns(v01, v02, co, si)
        v11, v12 = _mix_columns(v11, v12, co, si)
        v21, v22 = _mix_columns(v21, v22, co, si)
    return (
        ((a00, a01, a02), (a10, a11, a12), (a20, a21, a22)),
        ((v00, v01, v02), (v10, v11, v12), (v20, v21, v22)),
    )


@numba.njit(cache=True)
def _jacobi(tensor: Matrix3, pivot_at_top: bool) -> tuple[Matrix3, Matrix3]:
    """Returns the diagonalised tensor and its unit eigenvectors (as columns).

    `pivot_at_top` reproduces the stress loop, which tests the previous pivot and so
    always applies one extra rotation (KNOWN_ISSUES T4).
    """
    eigenvectors: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if pivot_at_top:
        pivot = 1.0
        while pivot > _EIGEN_TOLERANCE:
            pivot, big, small = _pick_pivot(tensor)
            tensor, eigenvectors = _rotate(tensor, eigenvectors, big, small)
    else:
        pivot, big, small = _pick_pivot(tensor)
        while pivot > _EIGEN_TOLERANCE:
            tensor, eigenvectors = _rotate(tensor, eigenvectors, big, small)
            pivot, big, small = _pick_pivot(tensor)
    return tensor, _normalise_columns(eigenvectors)


@numba.njit(cache=True)
def _normalise_columns(matrix: Matrix3) -> Matrix3:
    """Divides each non-zero column by its length."""
    c0 = _normalise(_column(matrix, 0))
    c1 = _normalise(_column(matrix, 1))
    c2 = _normalise(_column(matrix, 2))
    return ((c0[0], c1[0], c2[0]), (c0[1], c1[1], c2[1]), (c0[2], c1[2], c2[2]))


@numba.njit(cache=True)
def _normalise(vector: Vector) -> Vector:
    """Returns the vector divided by its length, or unchanged if zero."""
    length = norm(vector)
    return divide(vector, length) if length > 0 else vector
