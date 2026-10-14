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

STRAIN_START = 8
STRAIN_PERPENDICULAR_START = 26
STRESS_START = 4
STRESS_SECOND_START = 31
NORMAL_START = 32
