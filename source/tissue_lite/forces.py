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

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.material import (
    Material,
    compute_anisotropic_stress,
    compute_fibre_direction,
    compute_lame_parameters,
    compute_moduli,
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


@numba.njit(cache=True)
def compute_trbs_forces(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Adds the TRBS-MT forces to `vertex_forces` (V, 3) and `centre_forces` (C, 3).

    Both arrays are accumulated into, not zeroed. Reads `cells.young_longitudinal`,
    `concentration` and `microtubule`; a vanishing fibre direction consumes `rand_state`
    (KNOWN_ISSUES T1). Without softening, also writes each cell's transverse modulus
    into `cells.young_transverse` (Tissue's slot 20, T7); with softening it is left
    alone (C3). A triangle with a rest angle whose cosine is within 1e-4 of 1 gets
    Tissue's sliver "fix" in place of its own forces (T3).

    Raises:
        ZeroDivisionError: If a rest triangle is degenerate (a zero or straight angle).
    """
    for c in range(mesh.offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(
            cells.young_longitudinal[c], cells.concentration[c], material
        )
        if not material.soften:
            cells.young_transverse[c] = young_t
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
            # Delingette's springs of the transverse material.
            cosines = _compute_rest_cosines(rest_lengths)
            tensile, angular = _compute_spring_stiffness(
                cosines, rest_area, lame.lambda_t, lame.mu_t
            )
            spring = _compute_spring_forces(triangle, rest_lengths, tensile, angular)
            # The fibre stress, pushed onto the nodes through the shape vectors.
            fibre = compute_fibre_direction(
                gradient, triangle.x_axis, triangle.y_axis, microtubule, rand_state
            )
            stress = compute_anisotropic_stress(
                compute_green_strain(gradient),
                fibre,
                lame.lambda_l - lame.lambda_t,
                lame.mu_l - lame.mu_t,
            )
            fibre_forces = _compute_fibre_forces(
                triangle, rest_area, gradient, shape, stress
            )
            a = mesh.vertex_a[t]
            b = mesh.vertex_b[t]
            if _is_sliver(cosines):
                _double_accumulated_forces(c, a, b, vertex_forces, centre_forces)
            else:
                _add_node_forces(
                    c, a, b, spring, fibre_forces, vertex_forces, centre_forces
                )


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
def _add_node_forces(
    c: int,
    a: int,
    b: int,
    spring: NodeForces,
    fibre: NodeForces,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Accumulates spring + fibre forces onto the centre and the two vertices."""
    centre_forces[c, 0] += spring[0][0] + fibre[0][0]
    centre_forces[c, 1] += spring[0][1] + fibre[0][1]
    centre_forces[c, 2] += spring[0][2] + fibre[0][2]
    vertex_forces[a, 0] += spring[1][0] + fibre[1][0]
    vertex_forces[a, 1] += spring[1][1] + fibre[1][1]
    vertex_forces[a, 2] += spring[1][2] + fibre[1][2]
    vertex_forces[b, 0] += spring[2][0] + fibre[2][0]
    vertex_forces[b, 1] += spring[2][1] + fibre[2][1]
    vertex_forces[b, 2] += spring[2][2] + fibre[2][2]


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


@numba.njit(cache=True)
def compute_pressure_forces(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    pressure: float,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Adds `pressure * area / 2` along each triangle's normal to all three nodes.

    `pressure` is the current, ramped value. The consistent share would be area / 3
    (KNOWN_ISSUES P1). Accumulates into `vertex_forces` and `centre_forces`; the normal
    follows the cyclic vertex order of `Topology`.
    """
    offsets = mesh.offsets
    for c in range(offsets.shape[0] - 1):
        for t in range(offsets[c], offsets[c + 1]):
            triangle = compute_triangle_geometry(mesh, centres, vertices, c, t)
            coeff = pressure * (compute_triangle_area(triangle.lengths) / 2)
            for d in range(3):
                centre_forces[c, d] += coeff * triangle.normal[d]
                vertex_forces[mesh.vertex_a[t], d] += coeff * triangle.normal[d]
                vertex_forces[mesh.vertex_b[t], d] += coeff * triangle.normal[d]
