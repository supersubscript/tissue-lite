"""Forces on vertices and cell centres: TRBS-MT membrane and turgor pressure.

Ports the derivative parts of `VertexFromTRBScenterTriangulationMT` and
`Pressure3D::CenterTriangulation::Linear` from Tissue e74e457. Each centre
triangle contributes Delingette's biquadratic springs for the transverse (matrix)
material plus the fibre stress pushed onto the nodes through the shape vectors.
Every expression keeps Tissue's evaluation order so that results are
bit-identical; do not simplify them algebraically.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, Int8Array, IntArray
from tissue_lite.material import (
    Lame,
    Material,
    compute_anisotropic_stress,
    compute_fibre_direction,
    compute_lame_parameters,
    compute_moduli,
    find_fibre_direction,
)
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    Matrix2,
    ShapeVectors,
    TriangleGeometry,
    TriangleMesh,
    Vector,
    combine,
    compute_cosine,
    compute_deformation,
    compute_green_strain,
    compute_triangle_area,
    compute_triangle_geometry,
    get_rest_lengths,
    multiply,
    subtract,
)

type NodeForces = tuple[Vector, Vector, Vector]  # on the centre, vertex a, vertex b

_SLIVER = 0.0001


_NORMAL = 0
_SLIVER_TRIANGLE = 1
_RANDOM_FIBRE = 2


@numba.njit(cache=True)
def compute_forces(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
    pressure: float,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Adds the TRBS-MT forces, then the pressure forces, to `vertex_forces` (V, 3) and
    `centre_forces` (C, 3).

    Both arrays are accumulated into, not zeroed. Reads `cells.young_longitudinal`,
    `concentration` and `microtubule`; a vanishing fibre direction consumes `rand_state`
    (KNOWN_ISSUES T1). Without softening, also writes each cell's transverse modulus
    into `cells.young_transverse` (Tissue's slot 20, T7); with softening it is left
    alone (C3). A triangle with a rest angle whose cosine is within 1e-4 of 1 gets
    Tissue's sliver "fix" in place of its own forces (T3). `pressure` is the current,
    ramped value, applied as `pressure * area / 2` along each triangle's normal to all
    three nodes; the consistent share would be area / 3 (P1).

    The triangles are evaluated in parallel (Numba's thread count), then added up
    in Tissue's order with the random draws and slivers handled there, so the result
    does not depend on the thread count.

    Raises:
        ZeroDivisionError: If a rest triangle is degenerate (a zero or straight angle).
    """
    num_triangles = mesh.vertex_a.shape[0]
    node_forces = np.empty((num_triangles, 3, 3))
    pressure_forces = np.empty((num_triangles, 3))
    kinds = np.empty(num_triangles, dtype=np.int8)
    _compute_triangle_forces(
        mesh,
        centres,
        vertices,
        cells,
        material,
        pressure,
        node_forces,
        pressure_forces,
        kinds,
    )
    for c in range(mesh.offsets.shape[0] - 1):
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            a = mesh.vertex_a[t]
            b = mesh.vertex_b[t]
            if kinds[t] == _RANDOM_FIBRE:
                kinds[t] = _compute_random_fibre_triangle(
                    mesh,
                    centres,
                    vertices,
                    cells,
                    material,
                    rand_state,
                    c,
                    t,
                    node_forces[t],
                )
            if kinds[t] == _SLIVER_TRIANGLE:
                _double_accumulated_forces(c, a, b, vertex_forces, centre_forces)
            else:
                for d in range(3):
                    centre_forces[c, d] += node_forces[t, 0, d]
                for d in range(3):
                    vertex_forces[a, d] += node_forces[t, 1, d]
                for d in range(3):
                    vertex_forces[b, d] += node_forces[t, 2, d]
    for c in range(mesh.offsets.shape[0] - 1):
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            for d in range(3):
                centre_forces[c, d] += pressure_forces[t, d]
                vertex_forces[mesh.vertex_a[t], d] += pressure_forces[t, d]
                vertex_forces[mesh.vertex_b[t], d] += pressure_forces[t, d]


@numba.njit(cache=True, parallel=True)
def _compute_triangle_forces(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    pressure: float,
    node_forces: FloatArray,
    pressure_forces: FloatArray,
    kinds: Int8Array,
) -> None:
    """Writes each triangle's summed spring and fibre forces on its centre, vertex a
    and vertex b (T, 3, 3), its pressure force (T, 3) and its kind; triangles whose
    fibre direction needs a random draw are left to the caller."""
    # Numba drops writes through a NamedTuple field inside `prange`; write via a local.
    young_transverse = cells.young_transverse
    for c in numba.prange(mesh.offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(
            cells.young_longitudinal[c], cells.concentration[c], material
        )
        if not material.soften:
            young_transverse[c] = young_t
        lame = compute_lame_parameters(young_l, young_t, material)
        microtubule = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            triangle = compute_triangle_geometry(mesh, centres, vertices, c, t)
            rest_lengths = get_rest_lengths(mesh, t)
            rest_area, shape, gradient = compute_deformation(
                triangle.lengths, rest_lengths
            )
            found, f0, f1 = find_fibre_direction(
                gradient, triangle.x_axis, triangle.y_axis, microtubule
            )
            cosines = _compute_rest_cosines(rest_lengths)
            if not found:
                kinds[t] = _RANDOM_FIBRE
            elif _is_sliver(cosines):
                kinds[t] = _SLIVER_TRIANGLE
            else:
                kinds[t] = _NORMAL
                _write_node_forces(
                    triangle,
                    rest_lengths,
                    rest_area,
                    shape,
                    gradient,
                    cosines,
                    lame,
                    (f0, f1),
                    node_forces[t],
                )
            coeff = pressure * (compute_triangle_area(triangle.lengths) / 2)
            for d in range(3):
                pressure_forces[t, d] = coeff * triangle.normal[d]


@numba.njit(cache=True)
def _compute_random_fibre_triangle(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
    c: int,
    t: int,
    target: FloatArray,
) -> int:
    """Evaluates a triangle whose fibre direction is a random draw (KNOWN_ISSUES
    T1), drawing even for a sliver as Tissue does; returns its kind."""
    young_l, young_t = compute_moduli(
        cells.young_longitudinal[c], cells.concentration[c], material
    )
    lame = compute_lame_parameters(young_l, young_t, material)
    triangle = compute_triangle_geometry(mesh, centres, vertices, c, t)
    rest_lengths = get_rest_lengths(mesh, t)
    rest_area, shape, gradient = compute_deformation(triangle.lengths, rest_lengths)
    microtubule = (
        cells.microtubule[c, 0],
        cells.microtubule[c, 1],
        cells.microtubule[c, 2],
    )
    fibre = compute_fibre_direction(
        gradient, triangle.x_axis, triangle.y_axis, microtubule, rand_state
    )
    cosines = _compute_rest_cosines(rest_lengths)
    if _is_sliver(cosines):
        return _SLIVER_TRIANGLE
    _write_node_forces(
        triangle,
        rest_lengths,
        rest_area,
        shape,
        gradient,
        cosines,
        lame,
        fibre,
        target,
    )
    return _NORMAL


@numba.njit(cache=True)
def _write_node_forces(
    triangle: TriangleGeometry,
    rest_lengths: Vector,
    rest_area: float,
    shape: ShapeVectors,
    gradient: Matrix2,
    cosines: Vector,
    lame: Lame,
    fibre: tuple[float, float],
    target: FloatArray,
) -> None:
    """Writes spring + fibre force on the centre, vertex a and vertex b (rows of the
    (3, 3) target)."""
    # Delingette's springs of the transverse material.
    tensile, angular = _compute_spring_stiffness(
        cosines, rest_area, lame.lambda_t, lame.mu_t
    )
    spring = _compute_spring_forces(triangle, rest_lengths, tensile, angular)
    # The fibre stress, pushed onto the nodes through the shape vectors.
    stress = compute_anisotropic_stress(
        compute_green_strain(gradient),
        fibre,
        lame.lambda_l - lame.lambda_t,
        lame.mu_l - lame.mu_t,
    )
    fibre_forces = _compute_fibre_forces(triangle, rest_area, gradient, shape, stress)
    for node in range(3):
        for d in range(3):
            target[node, d] = spring[node][d] + fibre_forces[node][d]


@numba.njit(cache=True)
def _compute_rest_cosines(rest_lengths: Vector) -> Vector:
    """Cosines of the rest angles at the centre, vertex a and vertex b."""
    r0, r1, r2 = rest_lengths
    return (
        compute_cosine(r0, r2, r1),
        compute_cosine(r0, r1, r2),
        compute_cosine(r1, r2, r0),
    )


@numba.njit(cache=True)
def _is_sliver(cosines: Vector) -> bool:
    """True when a rest angle is within `_SLIVER` of 0 (Tissue tests only near 0)."""
    cos0, cos1, cos2 = cosines
    return abs(1 - cos0) < _SLIVER or abs(1 - cos1) < _SLIVER or abs(1 - cos2) < _SLIVER


@numba.njit(cache=True)
def _compute_spring_stiffness(
    cosines: Vector, rest_area: float, lambda_t: float, mu_t: float
) -> tuple[Vector, Vector]:
    """Delingette's tensile stiffness per edge and angular stiffness per node."""
    cos0, cos1, cos2 = cosines
    cot0 = cos0 / np.sqrt(1 - cos0 * cos0)
    cot1 = cos1 / np.sqrt(1 - cos1 * cos1)
    cot2 = cos2 / np.sqrt(1 - cos2 * cos2)
    tau = 1.0 / (rest_area * 16)
    lm = lambda_t + 2 * mu_t
    tensile = (
        (2 * cot2 * cot2 * lm + 2 * mu_t) * tau,
        (2 * cot0 * cot0 * lm + 2 * mu_t) * tau,
        (2 * cot1 * cot1 * lm + 2 * mu_t) * tau,
    )
    angular = (
        (2 * cot1 * cot2 * lm - 2 * mu_t) * tau,
        (2 * cot0 * cot2 * lm - 2 * mu_t) * tau,
        (2 * cot0 * cot1 * lm - 2 * mu_t) * tau,
    )
    return tensile, angular


@numba.njit(cache=True)
def _compute_spring_forces(
    triangle: TriangleGeometry, rest_lengths: Vector, tensile: Vector, angular: Vector
) -> NodeForces:
    """Spring forces on the three nodes (Tissue's `derIprim`, `derIIprim`)."""
    l0, l1, l2 = triangle.lengths
    r0, r1, r2 = rest_lengths
    ts0, ts1, ts2 = tensile
    as0, as1, as2 = angular
    dl0 = l0 * l0 - r0 * r0
    dl1 = l1 * l1 - r1 * r1
    dl2 = l2 * l2 - r2 * r2
    centre_a = ts0 * dl0 + as1 * dl1 + as0 * dl2
    centre_b = ts2 * dl2 + as2 * dl1 + as0 * dl0
    a_centre = ts0 * dl0 + as0 * dl2 + as1 * dl1
    a_b = ts1 * dl1 + as2 * dl2 + as1 * dl0
    b_centre = ts2 * dl2 + as0 * dl0 + as2 * dl1
    b_a = ts1 * dl1 + as1 * dl0 + as2 * dl2
    p0, p1, p2 = triangle.centre, triangle.vertex_a, triangle.vertex_b
    return (
        combine(centre_a, subtract(p1, p0), centre_b, subtract(p2, p0)),
        combine(a_centre, subtract(p0, p1), a_b, subtract(p2, p1)),
        combine(b_centre, subtract(p0, p2), b_a, subtract(p1, p2)),
    )


@numba.njit(cache=True)
def _compute_fibre_forces(
    triangle: TriangleGeometry,
    rest_area: float,
    gradient: Matrix2,
    shape: ShapeVectors,
    stress: Matrix2,
) -> NodeForces:
    """Fibre forces on the three nodes: the first Piola-Kirchhoff stress A_rest F S
    (Tissue's `TPK`) applied to each node's shape vector."""
    (p00, p01), (p10, p11) = multiply(gradient, stress)
    piola = (
        (rest_area * p00, rest_area * p01),
        (rest_area * p10, rest_area * p11),
    )
    return (
        _compute_node_force(triangle, piola, shape[0]),
        _compute_node_force(triangle, piola, shape[1]),
        _compute_node_force(triangle, piola, shape[2]),
    )


@numba.njit(cache=True)
def _compute_node_force(
    triangle: TriangleGeometry, piola: Matrix2, shape_vector: tuple[float, float]
) -> Vector:
    """Returns -(P · shape vector) brought back to global coordinates."""
    (p00, p01), (p10, p11) = piola
    sv0, sv1 = shape_vector
    force = combine(
        p00 * sv0 + p01 * sv1, triangle.x_axis, p10 * sv0 + p11 * sv1, triangle.y_axis
    )
    return -force[0], -force[1], -force[2]


@numba.njit(cache=True)
def _double_accumulated_forces(
    c: int, a: int, b: int, vertex_forces: FloatArray, centre_forces: FloatArray
) -> None:
    """Tissue's sliver "fix": adds the three nodes' accumulated forces to each of
    them instead of this triangle's own (KNOWN_ISSUES T3)."""
    for d in range(3):
        total = centre_forces[c, d] + vertex_forces[a, d] + vertex_forces[b, d]
        centre_forces[c, d] += total
        vertex_forces[a, d] += total
        vertex_forces[b, d] += total
