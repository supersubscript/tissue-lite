"""Per-cell stress and strain tensors, principal axes and energies.

Ports the `update` part of `VertexFromTRBScenterTriangulationMT` from Tissue
e74e457, including its Jacobi eigensolver. Every expression keeps Tissue's
evaluation order so that results are bit-identical; do not simplify them
algebraically.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.forces import RANDOM_THRESHOLD, random_direction
from tissue_lite.material import Material, compute_moduli
from tissue_lite.tissue import CellState
from tissue_lite.triangles import RestLengths, TriangleMesh

_EIGEN_TOLERANCE = 0.000001
_C45 = 0.70710678118


@numba.njit(cache=True)
def _pick_pivot(s: FloatArray) -> tuple[float, int, int]:
    """Returns the largest off-diagonal magnitude and its (I, J), I > J."""
    pivot, big, small = abs(s[1, 0]), 1, 0
    if abs(s[2, 0]) > pivot:
        pivot, big, small = abs(s[2, 0]), 2, 0
    if abs(s[2, 1]) > pivot:
        pivot, big, small = abs(s[2, 1]), 2, 1
    return pivot, big, small


@numba.njit(cache=True)
def _rotate(
    s: FloatArray, v: FloatArray, big: int, small: int, tmp: FloatArray
) -> None:
    """Applies Tissue's Jacobi rotation in the (small, big) plane.

    Skipping Tissue's exact products with 0 and 1 leaves the result bit-identical.
    """
    if abs(s[big, big] - s[small, small]) < _EIGEN_TOLERANCE:
        si = _C45
        co = _C45
    else:
        t = (2 * s[big, small]) / (s[small, small] - s[big, big])
        t = 1 / np.sqrt(1 + t * t)
        si = _C45 * np.sqrt(1 - t) if t < 1 else _C45 * np.sqrt(t - 1)
        co = _C45 * np.sqrt(1 + t)
    for r in range(3):
        a_lo = s[r, small]
        a_hi = s[r, big]
        tmp[r, small] = a_lo * co + a_hi * si
        tmp[r, big] = a_lo * -si + a_hi * co
        for q in range(3):
            if q != small and q != big:
                tmp[r, q] = s[r, q]
    for q in range(3):
        b_lo = tmp[small, q]
        b_hi = tmp[big, q]
        for r in range(3):
            if r == small:
                s[r, q] = co * b_lo + si * b_hi
            elif r == big:
                s[r, q] = -si * b_lo + co * b_hi
            else:
                s[r, q] = tmp[r, q]
    for r in range(3):
        v_lo = v[r, small]
        v_hi = v[r, big]
        v[r, small] = v_lo * co + v_hi * si
        v[r, big] = v_lo * -si + v_hi * co


@numba.njit(cache=True)
def _jacobi(s: FloatArray, v: FloatArray, pivot_at_top: bool, tmp: FloatArray) -> None:
    """Diagonalises s in place; columns of v become unit eigenvectors.

    `pivot_at_top` reproduces the stress loop, which tests the previous pivot and so
    always applies one extra rotation.
    """
    for r in range(3):
        for q in range(3):
            v[r, q] = 1.0 if r == q else 0.0
    if pivot_at_top:
        pivot = 1.0
        while pivot > _EIGEN_TOLERANCE:
            pivot, big, small = _pick_pivot(s)
            _rotate(s, v, big, small, tmp)
    else:
        pivot, big, small = _pick_pivot(s)
        while pivot > _EIGEN_TOLERANCE:
            _rotate(s, v, big, small, tmp)
            pivot, big, small = _pick_pivot(s)
    for q in range(3):
        norm = np.sqrt(v[0, q] * v[0, q] + v[1, q] * v[1, q] + v[2, q] * v[2, q])
        if norm > 0:
            for r in range(3):
                v[r, q] /= norm


@numba.njit(cache=True)
def _rotate_tensor(rot: FloatArray, local: FloatArray, tmp: FloatArray) -> None:
    """Replaces local by rot @ local @ rot.T, summing in Tissue's order."""
    for r in range(3):
        for q in range(3):
            acc = 0.0
            for w in range(3):
                acc += rot[r, w] * local[w, q]
            tmp[r, q] = acc
    for r in range(3):
        for q in range(3):
            acc = 0.0
            for w in range(3):
                acc += tmp[r, w] * rot[q, w]
            local[r, q] = acc


@numba.njit(cache=True)
def update_cell_stress(
    mesh: TriangleMesh,
    rest: RestLengths,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
) -> None:
    """Writes the area-weighted stress and strain of every cell into `cells`.

    Sets the tensors, normal, both principal axes of stress and strain, their
    anisotropies, the von Mises stress, rest area and energies, and normalises the
    microtubule axis, as Tissue's `update` does.
    """
    offsets, tri_v2, tri_v3, tri_wall, spoke_start, spoke_end, wall_vertices = mesh
    spokes, wall_lengths, extensions = rest
    x = vertices
    nu_l = material.poisson_longitudinal
    nu_t = material.poisson_transverse
    double = material.double_rest_lengths
    rot = np.empty((3, 3))
    tmp = np.empty((3, 3))
    strain_t = np.empty((3, 3))
    stress_t = np.empty((3, 3))
    strain_sum = np.empty((3, 3))
    stress_sum = np.empty((3, 3))
    v_stress = np.empty((3, 3))
    v_strain = np.empty((3, 3))
    normal = np.empty(3)
    for c in range(offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(cells, c, material)
        lam_l = young_l * nu_l / (1 - nu_l * nu_l)
        mio_l = young_l / (2 * (1 + nu_l))
        lam_t = young_t * nu_t / (1 - nu_t * nu_t)
        mio_t = young_t / (2 * (1 + nu_t))
        strain_sum[:] = 0.0
        stress_sum[:] = 0.0
        normal[:] = 0.0
        total_rest = 0.0
        total_area = 0.0
        energy_iso = 0.0
        energy_aniso = 0.0
        a_g0 = cells.microtubule[c, 0]
        a_g1 = cells.microtubule[c, 1]
        a_g2 = cells.microtubule[c, 2]
        for t in range(offsets[c], offsets[c + 1]):
            i2 = tri_v2[t]
            i3 = tri_v3[t]
            w = tri_wall[t]
            r0 = spokes[spoke_start[t]]
            r1 = wall_lengths[w] + extensions[t] if double else wall_lengths[w]
            r2 = spokes[spoke_end[t]]
            p00, p01, p02 = centres[c, 0], centres[c, 1], centres[c, 2]
            p10, p11, p12 = x[i2, 0], x[i2, 1], x[i2, 2]
            p20, p21, p22 = x[i3, 0], x[i3, 1], x[i3, 2]
            wa = wall_vertices[w, 0]
            wb = wall_vertices[w, 1]
            l0 = np.sqrt(
                (p00 - p10) * (p00 - p10)
                + (p01 - p11) * (p01 - p11)
                + (p02 - p12) * (p02 - p12)
            )
            dist = 0.0
            for d in range(3):
                dist += (x[wa, d] - x[wb, d]) * (x[wa, d] - x[wb, d])
            l1 = np.sqrt(dist)
            l2 = np.sqrt(
                (p00 - p20) * (p00 - p20)
                + (p01 - p21) * (p01 - p21)
                + (p02 - p22) * (p02 - p22)
            )
            rest_area = (
                np.sqrt(
                    (r0 + r1 + r2) * (-r0 + r1 + r2) * (r0 - r1 + r2) * (r0 + r1 - r2)
                )
                * 0.25
            )
            area = (
                np.sqrt(
                    (l0 + l1 + l2) * (-l0 + l1 + l2) * (l0 - l1 + l2) * (l0 + l1 - l2)
                )
                * 0.25
            )
            ca = (l0 * l0 + l1 * l1 - l2 * l2) / (l0 * l1 * 2)
            qa = ca * l0
            qc = np.sqrt(1 - ca * ca) * l0
            qb = l1
            ra = (r0 * r0 + r1 * r1 - r2 * r2) / (r0 * r1 * 2)
            pa = ra * r0
            pc = np.sqrt(1 - ra * ra) * r0
            pb = r1
            sv01 = 1 / pc
            sv21 = -pa / (pb * pc)
            sv20 = 1 / pb
            temp_a = np.sqrt(
                (p20 - p10) * (p20 - p10)
                + (p21 - p11) * (p21 - p11)
                + (p22 - p12) * (p22 - p12)
            )
            xc0 = (p20 - p10) / temp_a
            xc1 = (p21 - p11) / temp_a
            xc2 = (p22 - p12) / temp_a
            bc0 = (p00 - p10) / l0
            bc1 = (p01 - p11) / l0
            bc2 = (p02 - p12) / l0
            zc0 = xc1 * bc2 - xc2 * bc1
            zc1 = xc2 * bc0 - xc0 * bc2
            zc2 = xc0 * bc1 - xc1 * bc0
            zn = np.sqrt(zc0 * zc0 + zc1 * zc1 + zc2 * zc2)
            zc0 = zc0 / zn
            zc1 = zc1 / zn
            zc2 = zc2 / zn
            yc0 = zc1 * xc2 - zc2 * xc1
            yc1 = zc2 * xc0 - zc0 * xc2
            yc2 = zc0 * xc1 - zc1 * xc0
            rot[0, 0], rot[0, 1], rot[0, 2] = xc0, yc0, zc0
            rot[1, 0], rot[1, 1], rot[1, 2] = xc1, yc1, zc1
            rot[2, 0], rot[2, 1], rot[2, 2] = xc2, yc2, zc2
            al0 = xc0 * a_g0 + xc1 * a_g1 + xc2 * a_g2
            al1 = yc0 * a_g0 + yc1 * a_g1 + yc2 * a_g2
            d_lam = lam_l - lam_t
            d_mio = mio_l - mio_t
            f00 = qb * sv20
            f10 = 0.0
            f01 = qa * sv01 + qb * sv21
            f11 = qc * sv01
            ar0 = f11 * al0 - f01 * al1
            ar1 = -f10 * al0 + f00 * al1
            measure = np.sqrt(ar0 * ar0 + ar1 * ar1)
            if measure < RANDOM_THRESHOLD:
                ar0, ar1 = random_direction(rand_state)
            else:
                ar0 = ar0 / measure
                ar1 = ar1 / measure
            aa00 = ar0 * ar0
            aa01 = ar0 * ar1
            aa10 = ar1 * ar0
            aa11 = ar1 * ar1
            b00 = f00 * f00 + f01 * f01
            b01 = f00 * f10 + f01 * f11
            b10 = f10 * f00 + f11 * f01
            b11 = f10 * f10 + f11 * f11
            e00 = 0.5 * (f00 * f00 + f10 * f10 - 1)
            e01 = 0.5 * (f00 * f01 + f10 * f11)
            e10 = 0.5 * (f01 * f00 + f11 * f10)
            e11 = 0.5 * (f01 * f01 + f11 * f11 - 1)
            ee00 = e00 * e00 + e01 * e10
            ee01 = e00 * e01 + e01 * e11
            ee10 = e10 * e00 + e11 * e10
            ee11 = e10 * e01 + e11 * e11
            tr_e = e00 + e11
            i2_inv = ee00 + ee11
            i5 = ar0 * ar0 * ee00 + ar0 * ar1 * (ee01 + ee10) + ar1 * ar1 * ee11
            det_b = b00 * b11 - b10 * b01
            alm00 = 0.5 * (1 - (b11 / det_b))
            alm01 = 0.5 * b01 / det_b
            alm10 = 0.5 * b10 / det_b
            alm11 = 0.5 * (1 - (b00 / det_b))
            at_ea = ar0 * ar0 * e00 + ar0 * ar1 * (e01 + e10) + ar1 * ar1 * e11
            eaa00 = e00 * aa00 + e01 * aa10
            eaa01 = e00 * aa01 + e01 * aa11
            eaa10 = e10 * aa00 + e11 * aa10
            eaa11 = e10 * aa01 + e11 * aa11
            aae00 = aa00 * e00 + aa01 * e10
            aae01 = aa00 * e01 + aa01 * e11
            aae10 = aa10 * e00 + aa11 * e10
            aae11 = aa10 * e01 + aa11 * e11
            bb00 = b00 * b00 + b01 * b10
            bb01 = b00 * b01 + b01 * b11
            bb10 = b10 * b00 + b11 * b10
            bb11 = b10 * b01 + b11 * b11
            area_factor = rest_area / area
            g = lam_t * tr_e - mio_t
            sg00 = area_factor * (g * b00 + mio_t * bb00)
            sg01 = area_factor * (g * b01 + mio_t * bb01)
            sg10 = area_factor * (g * b10 + mio_t * bb10)
            sg11 = area_factor * (g * b11 + mio_t * bb11)
            ds00 = (d_lam / 2) * (tr_e * aa00 + at_ea) + d_mio * (eaa00 + aae00)
            ds01 = (d_lam / 2) * (tr_e * aa01) + d_mio * (eaa01 + aae01)
            ds10 = (d_lam / 2) * (tr_e * aa10) + d_mio * (eaa10 + aae10)
            ds11 = (d_lam / 2) * (tr_e * aa11 + at_ea) + d_mio * (eaa11 + aae11)
            dsf00 = ds00 * f00 + ds01 * f01
            dsf10 = ds10 * f00 + ds11 * f01
            dsf01 = ds00 * f10 + ds01 * f11
            dsf11 = ds10 * f10 + ds11 * f11
            dsg00 = area_factor * (f00 * dsf00 + f01 * dsf10)
            dsg10 = area_factor * (f10 * dsf00 + f11 * dsf10)
            dsg01 = area_factor * (f00 * dsf01 + f01 * dsf11)
            dsg11 = area_factor * (f10 * dsf01 + f11 * dsf11)
            strain_t[:] = 0.0
            stress_t[:] = 0.0
            strain_t[0, 0] = alm00
            strain_t[1, 0] = alm10
            strain_t[0, 1] = alm01
            strain_t[1, 1] = alm11
            stress_t[0, 0] = sg00 + dsg00
            stress_t[1, 0] = sg10 + dsg10
            stress_t[0, 1] = sg01 + dsg01
            stress_t[1, 1] = sg11 + dsg11
            _rotate_tensor(rot, strain_t, tmp)
            _rotate_tensor(rot, stress_t, tmp)
            for r in range(3):
                for q in range(3):
                    strain_sum[r, q] += area * strain_t[r, q]
            for r in range(3):
                for q in range(3):
                    stress_sum[r, q] += area * stress_t[r, q]
            normal[0] += area * zc0
            normal[1] += area * zc1
            normal[2] += area * zc2
            total_rest = total_rest + rest_area
            total_area = total_area + area
            energy_iso += ((lam_t / 2) * tr_e * tr_e + mio_t * i2_inv) * rest_area
            energy_aniso += ((d_lam / 2) * at_ea * tr_e + d_mio * i5) * rest_area
        for r in range(3):
            for q in range(3):
                strain_sum[r, q] = strain_sum[r, q] / total_area
        for r in range(3):
            for q in range(3):
                stress_sum[r, q] = stress_sum[r, q] / total_area
        for r in range(3):
            normal[r] /= total_area
        nn = np.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        if nn > 0:
            for r in range(3):
                normal[r] /= nn
        cells.stress_tensor[c] = stress_sum
        cells.strain_tensor[c] = strain_sum
        cells.normal[c, 0] = normal[0]
        cells.normal[c, 1] = normal[1]
        cells.normal[c, 2] = normal[2]

        _jacobi(stress_sum, v_stress, True, tmp)
        i_s = 0
        s1 = stress_sum[0, 0]
        if abs(stress_sum[1, 1]) > abs(s1):
            s1 = stress_sum[1, 1]
            i_s = 1
        if abs(stress_sum[2, 2]) > abs(s1):
            s1 = stress_sum[2, 2]
            i_s = 2
        if i_s == 0:
            i_s2, i_s3 = 1, 2
        elif i_s == 1:
            i_s2, i_s3 = 2, 0
        else:
            i_s2, i_s3 = 0, 1
        if abs(stress_sum[i_s3, i_s3]) > abs(stress_sum[i_s2, i_s2]):
            i_s2, i_s3 = i_s3, i_s2
        s2 = stress_sum[i_s2, i_s2]
        s3 = stress_sum[i_s3, i_s3]
        if (
            abs(
                normal[0] * v_stress[0, i_s]
                + normal[1] * v_stress[1, i_s]
                + normal[2] * v_stress[2, i_s]
            )
            > 0.7
        ):
            i_s = i_s2
            i_s2 = i_s3
            s1 = s2
            s2 = s3
        cells.von_mises_stress[c] = np.sqrt(s1 * s1 + s2 * s2 - s1 * s2)
        if abs(s1) < 0.000001:
            cells.stress_anisotropy[c] = 0
        else:
            cells.stress_anisotropy[c] = 1 - abs(s2 / s1)
        cells.stress_direction[c, 0] = v_stress[0, i_s]
        cells.stress_direction[c, 1] = v_stress[1, i_s]
        cells.stress_direction[c, 2] = v_stress[2, i_s]
        cells.stress_max[c] = s1
        cells.stress_direction_second[c, 0] = v_stress[0, i_s2]
        cells.stress_direction_second[c, 1] = v_stress[1, i_s2]
        cells.stress_direction_second[c, 2] = v_stress[2, i_s2]
        cells.stress_second[c] = s2

        _jacobi(strain_sum, v_strain, False, tmp)
        i_e = 0
        e1 = strain_sum[0, 0]
        if abs(strain_sum[1, 1]) > abs(e1):
            e1 = strain_sum[1, 1]
            i_e = 1
        if abs(strain_sum[2, 2]) > abs(e1):
            e1 = strain_sum[2, 2]
            i_e = 2
        if i_e == 0:
            i_e2, i_e3 = 1, 2
        elif i_e == 1:
            i_e2, i_e3 = 0, 2
        else:
            i_e2, i_e3 = 0, 1
        if abs(strain_sum[i_e3, i_e3]) > abs(strain_sum[i_e2, i_e2]):
            i_e2, i_e3 = i_e3, i_e2
        e2 = strain_sum[i_e2, i_e2]
        perp0 = normal[1] * v_strain[2, i_e] - normal[2] * v_strain[1, i_e]
        perp1 = normal[2] * v_strain[0, i_e] - normal[0] * v_strain[2, i_e]
        perp2 = normal[0] * v_strain[1, i_e] - normal[1] * v_strain[0, i_e]
        pn = np.sqrt(perp0 * perp0 + perp1 * perp1 + perp2 * perp2)
        if abs(pn) < 0.0001:
            perp0 = v_strain[0, i_e]
            perp1 = v_strain[1, i_e]
            perp2 = v_strain[2, i_e]
        if abs(e1) < 0.000001:
            cells.strain_anisotropy[c] = 0
        else:
            cells.strain_anisotropy[c] = 1 - abs(e2 / e1)
        cells.strain_direction[c, 0] = v_strain[0, i_e]
        cells.strain_direction[c, 1] = v_strain[1, i_e]
        cells.strain_direction[c, 2] = v_strain[2, i_e]
        cells.strain_max[c] = e1
        cells.strain_perpendicular[c, 0] = perp0
        cells.strain_perpendicular[c, 1] = perp1
        cells.strain_perpendicular[c, 2] = perp2
        cells.strain_second[c] = e2
        mt_norm = np.sqrt(a_g0 * a_g0 + a_g1 * a_g1 + a_g2 * a_g2)
        if mt_norm > 0:
            for d in range(3):
                cells.microtubule[c, d] = cells.microtubule[c, d] / mt_norm
        cells.rest_area[c] = total_rest
        cells.energy_isotropic[c] = energy_iso / total_rest
        cells.energy_anisotropic[c] = energy_aniso / total_rest
