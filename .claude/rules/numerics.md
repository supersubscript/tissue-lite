---
paths:
  - "source/**/*.py"
---

# Leverage libraries

If a one-liner using a well-known library function exists, use it.

- **Vectorise** — `np.interp`, `np.linspace`, `np.where`, broadcasting, pandas `.groupby` / `.transform` over Python loops.
- **scipy** for interpolation (`scipy.interpolate`), stats (`scipy.stats`), signal processing, and optimisation — don't hand-roll.
- **pandas idioms** — chained ops, `.explode()`, `.melt()`, `.pivot_table()` over row iteration.
- **numpy** for numerics — `np.interp`, `np.linalg`, `np.random`.

## Performance

- Vectorised ops over label images and tables; mind memory with large datasets.
- `tqdm` for progress in long-running computations.
