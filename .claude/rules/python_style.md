---
paths:
  - "**/*.py"
---

# Writing code
Match these templates. Prefer the good examples, avoid the bad.

Naming (function prefixes, variables, dataframes): [`python_naming_conventions.md`](python_naming_conventions.md).

## Docstrings

```python
# Public: explain behaviour the signature cannot.
def rank_by_distance(
    candidates: Sequence[str],
    target: str,
) -> list[str]:
    """Ranks candidates by edit distance from target.

    Args:
        candidates: Strings to rank.
        target: Reference string.

    Returns:
        Candidates sorted closest to target first.

    Raises:
        ValueError: If candidates is empty.
    """

# Private: concise one-line docstring.
def _compute_distance(a: str, b: str) -> int:
    """Returns the edit distance between two strings."""

# Good: adds semantic information.
def get_cell_labels(image: np.ndarray) -> np.ndarray:
    """Returns the sorted labels in image, without the background."""

# Bad: merely restates the signature.
def get_cell_labels(image: np.ndarray) -> np.ndarray:
    """Gets the cell labels of an image.

    Args:
        image: The image.

    Returns:
        The cell labels.
    """

# Good: document non-obvious behaviour, constraints, side effects and exceptions.
# Bad: repeat types, names, return values or implementation steps.
# Avoid empty Args/Returns/Raises sections and implementation narration.
```

## Implementation

```python
# Simple concise solutions, avoiding verbosity, and being precise in the implementation
# Bad - verbose
result = []
for x in items:
    if x % 2 == 0:
        result.append(x * 2)

# Good - target style
result = [x * 2 for x in items if x % 2 == 0]

# Good: precise typing.
def rank_candidates(
    candidates: Sequence[str],
    target: str,
) -> list[str]:
    ...

# Bad: missing or unnecessarily weak typing.
def rank_candidates(candidates: Any, target: Any) -> Any:
    ...
```

Also:
- Line length per `ruff format` (default 88). No walrus (`:=`).
- **Explicit signatures over config objects.** If multiple groups of parameters exist, consider grouping related parameters into a frozen dataclass.
- A docstring longer than the function it documents is too long. Docstrings should be as concise and succinct as possible.
- No comment block over 3 lines outside a module docstring.
- Leave existing comments and `# TODO` / `# FIXME` alone unless wrong or resolved.
- **No weak types** — `Any`, untyped `dict`/`list`, bare `object`, or blanket `# type: ignore` need a one-line justification.
- **No defensive try/except** — catch only at real boundaries (user input, external APIs, filesystem, network, subprocess), with a concrete strategy. Never `except Exception: pass`, never silent fallback.
- **Prefer an `if` guard to `try/except`** where a cheap check says the same thing.
- **Prefer built-in exceptions** — `ValueError`, `TypeError`, `KeyError`, `FileNotFoundError`.
- **One code path** — replacing something means deleting what it replaced, in the same PR.
- **No dead code** — grep for callers before deleting.
- **No circular imports** — extract the shared piece into a lower-level module.
- **DRY where it reduces complexity** — three similar lines are fine, the same block across modules is not.

# Project idioms — always
- Logging → `import logging; logger = logging.getLogger(__name__)`. `print` is forbidden in library code.
- Imports → absolute, from the concrete module path.
- Data models → frozen dataclasses; `Protocol` for structural interfaces; `TypedDict` only for narrow I/O shapes.
- Legacy file formats (`.init`, `.model`, solver files) are parsed in `tissue` and `legacy`; parsers raise `ValueError` on anything outside the supported subset.

# Testing
```python
@pytest.mark.parametrize(("label", "expected"), [(1, "L1"), (2, "L2"), (9, "L3+")])
def test_get_layer_names(label: int, expected: str) -> None:
    layers = [np.array([1]), np.array([2])]
    assert get_layer_names([label], layers)[0] == expected


def test_timepoint_rejects_other_names() -> None:
    with pytest.raises(ValueError, match="Not a timepoint"):
        TimepointId.parse("notes.csv")
```

- Flat layout: `tests/test_<module>.py`, functions `test_*`. Build small synthetic meshes and arrays inline; fidelity tests read only `tests/data/` through the `tests/conftest.py` fixtures.
- `uv run pytest [path]` is canonical.
- Fidelity tests compare against Tissue output with exact equality where the case is bit-exact; never loosen a tolerance to make a test pass.
- Long fidelity runs carry `@pytest.mark.slow` and are excluded by default.
- `@pytest.mark.parametrize` over loops for multi-case assertions.
- `pytest.raises(SpecificError)`, never bare `Exception`.
- Moving or renaming a module → grep `tests/` and call sites for the old path.

[pyproject.toml](../../pyproject.toml)
