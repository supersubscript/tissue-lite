"""Tissue's cell-variable slots, as used by every retained model file.

The legacy reader checks model files against these, so the magic indices live only
here.
"""

MICROTUBULE = slice(0, 3)
GROWTH_VALUE_1 = 3  # read by StrainTRBS growth (never written by any reaction)
STRESS_DIRECTION = slice(4, 7)
STRESS_MAX = 7
STRAIN_DIRECTION = slice(8, 11)
STRAIN_MAX = 11
YOUNG_LONGITUDINAL = 16
STRAIN_ANISOTROPY = 17
STRESS_ANISOTROPY = 18
REST_AREA = 19  # Tissue's "area ratio index"
YOUNG_TRANSVERSE = 20  # also the iso and aniso energy index
VON_MISES_STRESS = 21  # Tissue's "MT stress index"
VELOCITY = 22
ANGLE_MT_STRESS = 23
ANGLE_STRESS_STRAIN = 24
VOLUME_CHANGE = 25  # written in cells 1, 2 and 3
STRAIN_PERPENDICULAR = slice(26, 29)  # growth layout
STRAIN_SECOND = 29  # growth layout
GROWTH_COUNT = 30  # written by StrainTRBS growth
STRESS_DIRECTION_SECOND = slice(31, 34)  # growth layout (overwrites the normal)
STRESS_SECOND = 34  # growth layout
NORMAL = slice(32, 35)
STRESS_TENSOR_START = 26  # TRBS level-0 index; written only by dead branches

MICROTUBULE_START = MICROTUBULE.start
STRESS_START = STRESS_DIRECTION.start
STRAIN_START = STRAIN_DIRECTION.start
STRAIN_PERPENDICULAR_START = STRAIN_PERPENDICULAR.start
STRESS_SECOND_START = STRESS_DIRECTION_SECOND.start
NORMAL_START = NORMAL.start

# Where TRBS stores principal strain and stress axes: (strain starts, stress starts).
STORAGE = {
    "sweep": ([STRAIN_START], [STRESS_START]),
    "growth": (
        [STRAIN_START, STRAIN_PERPENDICULAR_START],
        [STRESS_START, STRESS_SECOND_START],
    ),
}

# TRBS level-0 indices: rest length, MT, strain and stress anisotropy, rest area,
# iso and aniso energy, longitudinal modulus, von Mises stress, stress tensor and
# normal.
TRBS_LEVEL0 = [
    0,
    MICROTUBULE_START,
    STRAIN_ANISOTROPY,
    STRESS_ANISOTROPY,
    REST_AREA,
    YOUNG_TRANSVERSE,
    YOUNG_TRANSVERSE,
    YOUNG_LONGITUDINAL,
    VON_MISES_STRESS,
    STRESS_TENSOR_START,
    NORMAL_START,
]
