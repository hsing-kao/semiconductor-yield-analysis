# Engineering Decision Log

This file records important analysis and modeling decisions, including the reasoning, alternatives considered, and limitations.

## Decision 1 — Preserve the original SECOM label definition

**Decision:**  
Use the original SECOM encoding during dataset inspection:

- `-1` = Pass
- `1` = Fail

Treat the failure class (`1`) as the positive class for failure-detection evaluation.

**Reason:**  
The label meaning was verified from the original `secom.names` documentation rather than inferred from the numerical values. Failure is the event of interest for recall, precision, F1, and related evaluation metrics.

**Alternative considered:**  
Re-encode Pass/Fail as `0/1` immediately.

**Why not yet:**  
Re-encoding is not necessary for dataset inspection. If later modeling code benefits from `0/1` encoding, the transformation will be explicit and documented.

---

## Decision 2 — Remove strictly constant features during preprocessing

**Decision:**  
Remove features with zero usable variation during model preprocessing.

**Observed during inspection:**  
116 of the 590 measurement features have at most one observed non-missing value.

**Reason:**  
A strictly constant feature cannot distinguish between pass and fail samples because it provides no variation across observations.

**Important implementation constraint:**  
Feature filtering used for modeling must be implemented in a leakage-safe way with respect to the training data rather than using validation outcomes to guide feature selection.

---

## Decision 3 — Inspect but do not automatically remove near-constant features

**Decision:**  
Define near-constant candidates during exploratory inspection as non-constant features with a dominant observed-value frequency of at least 98%. Do not automatically remove them from the MVP solely because they meet this definition.

**Observed during inspection:**  
10 features meet this exploratory criterion.

**Reason:**  
The observed dominant-frequency distribution showed a small group around 98.6%–99.9%, followed by a large drop to approximately 58%. The 98% threshold therefore provides a useful screening rule for this dataset.

A rare value may still contain predictive information, particularly because the SECOM features are anonymized. Low variation alone is not sufficient evidence that a non-constant feature is useless.

**Alternative considered:**  
Apply an arbitrary raw-variance threshold or automatically remove all features above the dominant-frequency threshold.

**Why not:**  
Raw variance depends on feature scale, and the physical units of the anonymous SECOM measurements are unknown. Automatic removal could also discard rare but potentially informative signals.

**Limitation:**  
The 98% threshold is an exploratory analysis choice, not a semiconductor process specification or universal statistical rule.
## Decision 4 — Keep downloaded raw data outside version control

**Decision:**  
Do not commit the downloaded UCI SECOM raw data files or generated processed datasets to the Git repository.

The following directories are excluded through `.gitignore`:

- `data/raw/`
- `data/processed/`

**Reason:**  
The SECOM dataset is an external data source rather than project-generated source code. Keeping downloaded raw data outside version control makes data provenance clearer and keeps the repository focused on reproducible code, documentation, and analysis decisions.

The project README should provide instructions for obtaining the dataset from UCI and placing the required files under `data/raw/`.

**Reproducibility implication:**  
A user who clones the repository cannot run the analysis immediately without first obtaining the SECOM dataset. Reproducibility therefore depends on documenting the external data source and expected local file structure.

**Alternative considered:**  
Commit the raw SECOM files directly into the repository.

**Why not:**  
The project can remain reproducible by documenting the external data source and download procedure without duplicating externally maintained raw data in the Git repository.