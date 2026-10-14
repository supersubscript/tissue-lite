# Project
flower-experiments: flower mechanics model. The Python package `flower` (in `source/flower/`: `curvature`, `mechanics`, `shape`, `colours`, `output`) analyses and plots simulation output; the legacy simulation drivers are in `source/scripts/simulation/`, are run directly and are not linted. `meshes/`, `models/` and `results/<run>/` hold the inputs and runs. Built with `uv`, `ruff`, and **ty**. Overview in `README.md`.

# Workflow
- **Deps**: `uv` (`uv.lock` committed). Never hand-edit `uv.lock`; use `uv add` (`--dev` for tools). Import packages that cannot be installed inside the functions that use them, and list them in `[tool.ty.analysis] allowed-unresolved-imports`.
- **Tasks**: `uv sync`, `uv run pytest [path]`, `uv run ruff check --fix`, `uv run ruff format`, `uv run ty check`.
- **Verify**: `uv run ruff check && uv run ty check && uv run pytest`; pre-commit runs ruff and ty on commit.
- **Python**: `>=3.14` via `requires-python`.
- **Layout**: all code lives under `source/`: the package in `source/<package>/`, standalone scripts in `source/scripts/`. No top-level `code/` folder.
- **Data**: `data/` and `figures/` are gitignored and large; never stage them.
- **Issues / PRs**: an issue states intent; a PR states what was done and how it was verified.

# Writing — commits, issues, PRs, descriptions, comments, docstrings, docs, replies
Prose in a PR description gets the same scrutiny as prose in a docstring.
- Answer first. Reasoning only where it changes a decision. No preamble, no restating the question, no recap unless asked.
- Cut every sentence carrying no information. Prefer a line to a paragraph, a table to prose.
- Concrete over abstract — numbers, paths, symbols. Not "significantly improved", "robust", "seamless".
- A PR or issue description that doesn't fit on one screen means the change is too big — split it, don't compress the description.
- No slop: binary contrasts ("it's not X, it's Y"), throat-clearing ("here's the thing"), faux-insight ("what nobody tells you"), colon reveals, dramatic fragments, puffery ("a testament to", "pivotal"), weasel attribution ("studies show"), fake-profound endings.

# Guides — read the relevant subset, don't guess
Read `README.md` before changing a module or script, and the docstring of the module you change.

Path-scoped rules in `.claude/rules/` load automatically when matching files are read:

| Task | Rules |
|------|-------|
| Python code | `python_style.md`, `python_naming_conventions.md` |
| Numerics / data analysis | `numerics.md` |

# Non-negotiables (always applicable)
- Logging → stdlib `logging` with a module-level logger (`logger = logging.getLogger(__name__)`). No `print` in library code.
- Imports → absolute only. Import from the concrete module path.
- Types → frozen dataclasses for records; parse filenames in a `parse` classmethod that raises `ValueError`. `Protocol` for structural interfaces; `TypedDict` only for narrow I/O shapes.
- Dataframes → name variables `df_<description>` (e.g. `df_results`), not `data` / `table`. Persist as tab-separated CSV with `index=False` unless another format is required.

# Python code
Before writing or editing Python, read `.claude/rules/python_style.md`: the worked
example of house style, the project idioms, and the testing rules. Function and variable naming: `.claude/rules/python_naming_conventions.md`.
Both load automatically on `*.py`.

# Minimum viable diff
Make the smallest change that satisfies the request.
- Every changed line traces to the request. No unrequested refactor, rename, or reformat.
- Prefer editing an existing file over adding one.
- No new abstraction (interface, base class, factory, wrapper, generic helper) under 3 real call sites.
- No caching, memoisation, or batching without a named performance requirement.
- Log at decision points and real boundaries only. Over-logging is a common agent failure mode.
- Match the surrounding style, naming, and patterns. Don't introduce a new one.
- 200 lines that could be 50 → rewrite it.
- More than ~20 lines of new code, or a new file → propose the plan first.

# Secrets & environment
- Settings → CLI arguments; no environment variables.
- Never commit secrets, API tokens, or imaging data.

# Ask before
Changing dependency versions or `uv.lock` · adding a dependency · deleting files · deleting or modifying an existing test (adding is fine) · rewriting files under `data/` in place · pushing or force-pushing.

# Large changes — interview, spec, fresh session
Sizeable = new module, cross-package refactor, new pipeline, migration, anything spanning many files and hours.
- Before coding, offer an `AskUserQuestion` interview: implementation trade-offs, CLI / config / API shape, edge cases and failure modes, anything not yet considered. Skip the obvious questions.
- Once it converges, write the design to `SPEC.md` and suggest a fresh session to execute it.
- Small fixes, single-file tweaks, and already-specified changes proceed directly.

# Persisting rules
Corrected on the same thing twice in a session → offer to make it stick. A short universal rule goes in this file (`.claude/rules/styleguide.md`); a multi-step workflow with arguments or side effects becomes a skill at `.claude/skills/<name>/SKILL.md` (`name` + `description` frontmatter, plus `disable-model-invocation: true` when it should only run on an explicit `/<name>`). Never edit this file or create a skill without confirmation.

[pyproject.toml](../../pyproject.toml)
