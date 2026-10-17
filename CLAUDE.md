# Project instructions

Shared coding rules live in `.claude/rules/` and load automatically: `styleguide.md` and
`collaboration-preferences.md` always; `python_style.md`, `python_naming_conventions.md` and
`numerics.md` when matching files are read. These are symlinks to the workspace copies; rules for
this repo alone go in `.claude/rules/repo.md`, which always loads. Edit those rather than this file.

Additionally:
- Do not author commits or issues as Claude.
- Never @-mention, tag, or request a review from anyone — in issues, PRs, descriptions, comments,
  or commit messages — unless the user explicitly names who to tag.
- Fidelity first: behaviour must match Tissue commit e74e457 bit for bit where the tests say so.
  Suspected bugs go in `KNOWN_ISSUES.md`, not in fixes.
