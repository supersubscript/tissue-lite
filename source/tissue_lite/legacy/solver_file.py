"""Reader for Tissue's RK5Adaptive solver files (`.rk5`)."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tissue_lite.solvers.cash_karp import CashKarp

_VTU_PRINT_FLAG = 2


@dataclass(frozen=True)
class SolverFile:
    """Contents of an RK5Adaptive solver file."""

    solver: CashKarp
    t_start: float
    t_end: float
    n_frames: int


def read_solver_file(path: Path) -> SolverFile:
    """Reads `RK5Adaptive t_start t_end flag n_print h1 eps`, ignoring `#` comments.

    `h1` becomes `CashKarp.max_step` and `eps` its tolerance. A print `flag` other than
    2 (VTU) logs a warning and the output stays VTU (KNOWN_ISSUES L4).

    Raises:
        ValueError: If it names another solver or its six values are missing or invalid.
    """
    tokens = [
        tok
        for line in path.read_text().splitlines()
        for tok in line.split("#")[0].split()
    ]
    if not tokens or tokens[0] != "RK5Adaptive":
        raise ValueError("Only RK5Adaptive solver files are supported")
    t_start, t_end, flag, n_print, h1, eps = tokens[1:7]
    if int(flag) != _VTU_PRINT_FLAG:
        logger.warning("Print flag {} is not supported; output is VTU (flag 2)", flag)
    return SolverFile(
        solver=CashKarp(max_step=float(h1), tolerance=float(eps)),
        t_start=float(t_start),
        t_end=float(t_end),
        n_frames=int(n_print),
    )
