---
paths:
  - "**/*.py"
---

# Naming scheme conventions

## How to use
New symbols should follow this naming scheme. Do not rename existing non-compliant names unless the task is the rename. If you spot one in passing, mention it as a non-actioned nit in the reply — do not change it in the diff.

## Functions
Use descriptive function prefixes that communicate the operation being performed. Prefer specific verbs over generic names such as `process_`, `handle_`, or `do_`.

### Retrieval, search, derivation and extraction

- `get_` — retrieve a value or object when its location/key/relationship is known, or when the derivation is trivial.
- `find_` — search for something matching a criterion; typically returns a (the first) match, or `None` if no match is found.
- `parse_` — interpret structured text or raw input into a structured representation.
- `extract_` — derive or pull information from a larger object or dataset, particularly when the operation is non-trivial.
- `compute_` — perform a substantive computational operation. Preferred alternative to `calculate_`.
- `infer_` — derive something that is not directly observed.

### Construction and generation

- `make_` — construct an object or value, usually as a convenience/factory function. Preferred alternative to `create_`.
- `build_` — assemble something from multiple components.
- `generate_` — produce new data or output, often algorithmically or synthetically.

### Machine learning

- `fit_` — optimise model parameters using data.
- `predict_` — produce model-based predictions.
- `score_` — calculate a score, ranking value, or similarity measure.
- `evaluate_` — assess model or system performance.

### Data manipulation
- `filter_` — retain elements satisfying a condition.
- `merge_` — combine compatible objects or datasets.
- `aggregate_` — combine values or groups according to an aggregation rule.
- `sample_` — select a sample from a population.
- `select_` — subset by identity, index, or an explicit member list.
- `convert_` — change the representation or type of something.

### Input/output

- `read_` — read raw or near-raw content from a source.
- `load_` — load data or an object from persistent/external storage.
- `write_` — write text data to a destination.
- `save_` — persist an object or result.

### Validation and predicates

- `validate_` — validate an object against rules or constraints.
- `is_` / `has_` / `can_` — boolean predicates; should generally return `bool`.

## Variable naming conventions

Images carry their role: `seg_img` for a label image, `seg_path` for its file, `resolution` for (z, y, x) voxel sizes in micrometres. Cell identities are `label`s; `BACKGROUND` (0) is not a cell.

```
# Good: expressive domain names.
neighbours
sister_pairs
match_scores
score_threshold
max_cells

# Bad: too generic, abbreviated, or single-letter names
# outside tight numeric scopes.
x
img
argument_1
tmp

# Good: dataframe naming indicates type and contained information.
df_tracking
df_pair_features

# Bad: dataframe naming is vague and inconsistent.
data
csv_info
retrieved_df
```
