"""Reader for Tissue's `.model` files, limited to the retained reactions.

Anything outside the retained subset, or using cell-variable slots other than the
ones in `legacy.slots`, is rejected rather than silently reinterpreted.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tissue_lite.feedback import (
    COUPLING_CODES,
    FibreStiffness,
    Initialisation,
    MicrotubuleAlignment,
)
from tissue_lite.growth import StrainGrowth
from tissue_lite.legacy import slots
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.material import (
    Boundary,
    ConcentrationSoftening,
    Pressure,
    TRBSMembrane,
)
from tissue_lite.model import MeristemModel

_CANONICAL_ORDER = (
    "FiberModel",
    "CenterTriangulation::Initiate",
    "VertexFromTRBScenterTriangulationMT",
    "Pressure3D::CenterTriangulation::Linear",
    "UpdateMTDirectionEquilibrium",
    "CenterTriangulation::WallGrowth::StrainTRBS",
    "CalculateAngleVectors",
    "CalculateAngleVectors",
    "TemplateVolumeChange",
    "VertexNoUpdateBoundary",
    "Calculate::VertexVelocity",
)
_REQUIRED = {
    "CenterTriangulation::Initiate",
    "VertexFromTRBScenterTriangulationMT",
    "Pressure3D::CenterTriangulation::Linear",
    "CalculateAngleVectors",
    "TemplateVolumeChange",
    "VertexNoUpdateBoundary",
}
_IGNORED_DIRECTIONS = {"StaticDirection", "ParallellDirection"}
_FIBRE_CODES = {code: name for name, code in COUPLING_CODES.items()}
_INITIALISE: dict[int, Initialisation] = {0: "none", 1: "isotropic", 2: "anisotropic"}

# TRBS level-0 indices: rest length, MT, strain and stress anisotropy, rest area,
# iso and aniso energy, longitudinal modulus, von Mises stress, stress tensor, normal.
_TRBS_LEVEL0 = [
    0,
    slots.MICROTUBULE.start,
    slots.STRAIN_ANISOTROPY,
    slots.STRESS_ANISOTROPY,
    slots.REST_AREA,
    slots.YOUNG_TRANSVERSE,
    slots.YOUNG_TRANSVERSE,
    slots.YOUNG_LONGITUDINAL,
    slots.VON_MISES_STRESS,
    slots.STRESS_TENSOR_START,
    slots.NORMAL.start,
]
# (strain starts, stress starts) where TRBS stores the principal axes → second axes.
_STORAGE: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] = {
    ((slots.STRAIN_DIRECTION.start,), (slots.STRESS_DIRECTION.start,)): False,
    (
        (slots.STRAIN_DIRECTION.start, slots.STRAIN_PERPENDICULAR.start),
        (slots.STRESS_DIRECTION.start, slots.STRESS_DIRECTION_SECOND.start),
    ): True,
}


@dataclass(frozen=True)
class Reaction:
    """One reaction block of a model file: parameters and one index list per level."""

    name: str
    parameters: list[float]
    indices: list[list[int]]


@dataclass(frozen=True)
class LegacyModel:
    """A model file's content: the model and the slot numbering it addresses."""

    model: MeristemModel
    layout: SlotLayout


def _split_tokens(text: str) -> list[str]:
    """Splits a Tissue file into tokens, dropping `#` comments."""
    return [tok for line in text.splitlines() for tok in line.split("#")[0].split()]


def parse_reactions(text: str) -> list[Reaction]:
    """Parses the reaction blocks of a model file, dropping its direction rules.

    `StaticDirection` and `ParallellDirection` are no-ops without division
    (KNOWN_ISSUES L3).

    Raises:
        ValueError: If the file is truncated, has compartment changes or other
            directions.
    """
    tokens = _split_tokens(text)
    pos = 0

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Model file ended early")
        pos += 1
        return tokens[pos - 1]

    n_reactions, n_compartment, n_direction = int(take()), int(take()), int(take())
    if n_compartment:
        raise ValueError("Compartment changes (e.g. division) are not supported")
    reactions = []
    for _ in range(n_reactions + n_direction):
        name = take()
        n_par, n_levels = int(take()), int(take())
        sizes = [int(take()) for _ in range(n_levels)]
        parameters = [float(take()) for _ in range(n_par)]
        indices = [[int(take()) for _ in range(n)] for n in sizes]
        reactions.append(Reaction(name=name, parameters=parameters, indices=indices))
    directions = reactions[n_reactions:]
    unknown = {d.name for d in directions} - _IGNORED_DIRECTIONS
    if unknown:
        raise ValueError(f"Unsupported direction rules: {sorted(unknown)}")
    return reactions[:n_reactions]


def read_model_file(path: Path) -> LegacyModel:
    """Reads a Tissue model file; raises ValueError outside the retained subset."""
    return build_model(parse_reactions(path.read_text()))


def build_model(reactions: list[Reaction]) -> LegacyModel:
    """Maps retained Tissue reactions onto a `MeristemModel` and its slot layout.

    Kept as in Tissue: MF=10 reads K and n from p9 and p10 (KNOWN_ISSUES C1), p9 = 1 is
    rejected (C2) and p10 == 1 selects double rest lengths (T11); FiberModel keeps its
    own moduli, with a warning (F2); without `Calculate::VertexVelocity`,
    `track_velocity` is off (F3).

    Raises:
        ValueError: If a reaction, its order, its parameters or its cell-variable slots
            fall outside the supported subset (L1).
    """
    _validate_order([r.name for r in reactions])
    by_name = {r.name: r for r in reactions}
    trbs = by_name["VertexFromTRBScenterTriangulationMT"]
    membrane, layout, com = _parse_membrane(trbs)
    initiate = by_name["CenterTriangulation::Initiate"]
    double = len(initiate.parameters) == 2 and initiate.parameters[1] == 1
    if double != (trbs.parameters[10] == 1):
        raise ValueError("Initiate double-edge flag and TRBS p10 disagree")
    if not double and "CenterTriangulation::WallGrowth::StrainTRBS" in by_name:
        raise ValueError("StrainTRBS growth needs double rest lengths (T11)")
    if initiate.indices != [[com]]:
        raise ValueError("Initiate and TRBS centre indices disagree")
    _validate_angles([r for r in reactions if r.name == "CalculateAngleVectors"])
    volume = [1, slots.VOLUME_CHANGE, 2, slots.VOLUME_CHANGE, 3, slots.VOLUME_CHANGE]
    if by_name["TemplateVolumeChange"].indices != [volume]:
        raise ValueError("TemplateVolumeChange must store into cells 1-3, slot 25")
    model = MeristemModel(
        membrane=membrane,
        pressure=_parse_pressure(
            by_name["Pressure3D::CenterTriangulation::Linear"], com
        ),
        boundary=_parse_boundary(by_name["VertexNoUpdateBoundary"]),
        fibres=_parse_fibres(by_name["FiberModel"], membrane)
        if "FiberModel" in by_name
        else None,
        microtubules=_parse_microtubules(by_name["UpdateMTDirectionEquilibrium"])
        if "UpdateMTDirectionEquilibrium" in by_name
        else None,
        growth=_parse_growth(
            by_name["CenterTriangulation::WallGrowth::StrainTRBS"], com
        )
        if "CenterTriangulation::WallGrowth::StrainTRBS" in by_name
        else None,
        track_velocity=_has_velocity(by_name.get("Calculate::VertexVelocity")),
    )
    return LegacyModel(model=model, layout=layout)


def _validate_order(names: list[str]) -> None:
    """Checks the reactions are a supported subsequence of the canonical order."""
    unknown = set(names) - set(_CANONICAL_ORDER)
    if unknown:
        raise ValueError(f"Unsupported reactions: {sorted(unknown)}")
    missing = _REQUIRED - set(names)
    if missing:
        raise ValueError(f"Missing required reactions: {sorted(missing)}")
    if names.count("CalculateAngleVectors") != 2:
        raise ValueError("Expected exactly two CalculateAngleVectors")
    rest = iter(_CANONICAL_ORDER)
    if not all(name in rest for name in names):
        raise ValueError(f"Reactions are not in the supported order: {names}")


def _parse_membrane(r: Reaction) -> tuple[TRBSMembrane, SlotLayout, int]:
    """Returns the membrane, the slot layout and the centre index."""
    p = r.parameters
    if len(p) not in (11, 13):
        raise ValueError("TRBS-MT needs 11 parameters (13 for MF=10)")
    mf = p[4]
    if mf not in (1, 10):
        raise ValueError(f"TRBS-MT material flag {mf} is not supported (1 or 10)")
    if p[5] != 0 or p[6] != 0 or p[7] != 1 or p[8] != 0:
        raise ValueError("TRBS-MT needs p5=0, p6=0, p7=1 (plane stress) and p8=0")
    if p[9] not in (0, 2, 3, 4):
        raise ValueError("TRBS-MT p9 must be 0, 2, 3 or 4 (1 selects the TETA branch)")
    if len(r.indices) != 4 or len(r.indices[1]) != 1:
        raise ValueError("TRBS-MT needs four index levels")
    level0 = r.indices[0]
    softening = None
    concentration_slot = None
    if mf == 10:
        if len(level0) != 12:
            raise ValueError("MF=10 needs a 12th level-0 index (the concentration)")
        # Tissue reads K and n from p9 and p10, not from p11 and p12 (KNOWN_ISSUES C1).
        softening = ConcentrationSoftening(k_hill=p[9], n_hill=p[10])
        if (p[11], p[12]) != (p[9], p[10]):
            logger.warning(
                "MF=10 uses K = p9 = {:g} and n = p10 = {:g} as Tissue does; "
                "p11 = {:g} and p12 = {:g} are ignored (KNOWN_ISSUES C1)",
                p[9],
                p[10],
                p[11],
                p[12],
            )
        concentration_slot = level0[11]
        level0 = level0[:11]
    if level0 != _TRBS_LEVEL0:
        raise ValueError(f"TRBS-MT level-0 indices must be {_TRBS_LEVEL0}")
    storage = (tuple(r.indices[2]), tuple(r.indices[3]))
    if storage not in _STORAGE:
        raise ValueError(f"Strain/stress storage must be one of {list(_STORAGE)}")
    membrane = TRBSMembrane(
        young_matrix=p[0],
        young_fibre=p[1],
        poisson_longitudinal=p[2],
        poisson_transverse=p[3],
        softening=softening,
    )
    layout = SlotLayout(
        second_axes=_STORAGE[storage], concentration_slot=concentration_slot
    )
    return membrane, layout, r.indices[1][0]


def _parse_pressure(r: Reaction, com: int) -> Pressure:
    """Validates the pressure reaction against the TRBS centre index."""
    p = r.parameters
    if len(p) != 3 or p[1] != 1:
        raise ValueError("Pressure needs three parameters with area flag 1")
    if r.indices != [[com]]:
        raise ValueError("Pressure centre index disagrees with TRBS")
    return Pressure(pressure=p[0], ramp_time=p[2])


def _parse_boundary(r: Reaction) -> Boundary:
    """Returns the fixed axes (all three when the reaction has no indices)."""
    if not r.indices:
        return Boundary()
    return Boundary(fixed_axes=tuple(r.indices[0]))


def _parse_fibres(r: Reaction, membrane: TRBSMembrane) -> FibreStiffness:
    """Maps FiberModel, keeping its own moduli when they differ from TRBS."""
    p = r.parameters
    if len(p) != 8:
        raise ValueError("FiberModel needs eight parameters")
    same = (p[5], p[6]) == (membrane.young_matrix, membrane.young_fibre)
    if not same:
        logger.warning(
            "FiberModel moduli ({:g}, {:g}) differ from the TRBS ones", p[5], p[6]
        )
    if r.indices[0] != [slots.STRESS_ANISOTROPY] or r.indices[2] != [slots.VELOCITY]:
        raise ValueError("FiberModel must read slots 18 and 22")
    if r.indices[1][0] != slots.YOUNG_LONGITUDINAL:
        raise ValueError("FiberModel must write slot 16")
    if int(p[2]) not in _FIBRE_CODES or int(p[7]) not in _INITIALISE:
        raise ValueError("Unknown FiberModel coupling or initialisation flag")
    return FibreStiffness(
        rate=p[0],
        coupling=_FIBRE_CODES[int(p[2])],
        k_hill=p[3],
        n_hill=p[4],
        velocity_threshold=p[1],
        initialise=_INITIALISE[int(p[7])],
        young_matrix=None if same else p[5],
        young_fibre=None if same else p[6],
    )


def _parse_microtubules(r: Reaction) -> MicrotubuleAlignment:
    """Maps UpdateMTDirectionEquilibrium; absent parameters disable its gates."""
    p = r.parameters
    expected = [
        [slots.STRESS_DIRECTION.start],
        [slots.MICROTUBULE.start],
        [slots.VELOCITY],
        [slots.STRESS_MAX, slots.VON_MISES_STRESS],
    ][: len(r.indices)]
    if r.indices != expected or len(r.indices) < 2:
        raise ValueError(f"UpdateMTDirectionEquilibrium indices must be {expected}")
    if len(p) >= 3 and len(r.indices) < 4:
        raise ValueError("Stress-difference threshold needs the fourth index level")
    return MicrotubuleAlignment(
        rate=p[0],
        velocity_threshold=p[1] if len(p) >= 2 else None,
        stress_difference_threshold=p[2] if len(p) >= 3 else None,
    )


def _parse_growth(r: Reaction, com: int) -> StrainGrowth:
    """Maps StrainTRBS growth; its velocity threshold is unused, as in Tissue."""
    p = r.parameters
    if len(p) != 4 or p[3] != 1:
        raise ValueError("StrainTRBS growth needs four parameters with p3=1")
    if r.indices != [[0], [com], [slots.VELOCITY]]:
        raise ValueError("StrainTRBS indices must be [0], [centre], [22]")
    logger.debug("StrainTRBS velocity threshold {:g} is unused, as in Tissue", p[2])
    return StrainGrowth(rate=p[0], threshold=p[1])


def _has_velocity(r: Reaction | None) -> bool:
    """Whether the model tracks vertex velocity (in slot 22 only)."""
    if r is None:
        return False
    if r.indices != [[slots.VELOCITY]]:
        raise ValueError("Calculate::VertexVelocity must store into slot 22")
    return True


def _validate_angles(angles: list[Reaction]) -> None:
    """Checks both CalculateAngleVectors use the standard slots."""
    expected = [
        [
            [slots.MICROTUBULE.start, slots.STRESS_DIRECTION.start],
            [slots.ANGLE_MT_STRESS],
        ],
        [
            [slots.STRESS_DIRECTION.start, slots.STRAIN_DIRECTION.start],
            [slots.ANGLE_STRESS_STRAIN],
        ],
    ]
    if [a.indices for a in angles] != expected:
        raise ValueError(f"CalculateAngleVectors indices must be {expected}")
