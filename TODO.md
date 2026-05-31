# TODO

## Review & push feature branches for PR

- [x] **`cv-bandwidth-auto`** (commit `2b9d581` — "add .fit() for auto bandwidth detection")
  - Pushed to `wolf/cv-bandwidth-auto`
  - Adds `.fit(X, y)` to `LocalBootstrap` / `LocalPermutation`; enables `bandwidth='auto'` / `k='auto'`
  - Open PR against `main`

- [X] **`ballkfold`** (commit `be08781` — "initial BallKFold using map coloring")
  - **Not yet pushed** — needs `git push -u wolf ballkfold`
  - Adds `BallKFold` (spatially exclusive folds via `mapclassify.greedy`)
  - Renames `DispersionKFold` -> `HilbertKFold` (file, class, tests, notebook)
  - Open PR against `main`
  - Heads up: `tests/test_range.py::test_correlogram_range_as_threshold` still uses removed `LocalPermutation(threshold=...)` -- pre-existing, but will fail in CI

- [x] **`metrics`** (branch `metrics`, unstaged — `_gearygram.py`, `CellStratifiedKFold`)
  - Not committed — awaiting user review before committing
  - Adds `gearygram`: Geary's C correlogram in three modes (bandwidth, kNN, LOWESS nonparametric); multivariate via Anselin (2019)
  - Adds `CellStratifiedKFold`: DGGS-stratified k-fold using H3 / A5 / HEALPix / S2 tiles
  - `examples/gearygram.ipynb` executed and symlinked to `docs/source/examples/`; wired into `docs/source/index.md`, `api.rst`, `references.bib`
  - `statsmodels>=0.14` added to all three CI/docs environment files
  - Known follow-ups: export `CellStratifiedKFold` from `cv/__init__.py` and top-level `__init__.py`; `LeaveBallOut` also unstaged and awaiting review

- [x] **`clusterkfold`** (commit `7f8dea8` — "initial ClusterStratifiedKFold using sklearn clusterer")
  - **Not yet pushed** — needs `git push -u wolf clusterkfold`
  - validate cluster stratified k fold

- [x] **`metrics`** (commit `0cfd24f` — "initial metrics module with area_of_applicability")
  - **Not yet pushed** — needs `git push -u wolf metrics`
  - Adds `geovalidate.metrics` subpackage with `area_of_applicability` (Meyer & Pebesma 2021), reworked from nanophyto/abil `analyze.py`
  - Includes `examples/aoa_vs_cast.qmd` (Quarto) comparing geovalidate vs `CAST::aoa` via reticulate — DI Pearson r = 1.0000, AOA mask agreement 100%
  - Heads up: rendered html (`examples/aoa_vs_cast.html`) is left untracked; regenerate with `QUARTO_R=/Library/Frameworks/R.framework/Resources/R RETICULATE_PYTHON=~/miniforge/envs/geovalidate/bin/python quarto render examples/aoa_vs_cast.qmd`
  - Heads up: the rework fixed several bugs in the original abil implementation (Tukey percentile scale, float-threshold scale, CV-split unpacking, polarity flip) — see commit message in `_aoa.py` FIXED-relative-to comments
  - Possible follow-ups before PR: re-introduce a clean local-point-density metric; add NaN-masking back as an optional flag; consider adding a sklearn-style `AreaOfApplicability` class wrapping the function

- [x] **`main`** (commit `4892ff0` -- "add Sphinx docs and wire ClusterStratifiedKFold into public API")
  - Not pushed
  - Adds full Sphinx documentation stack (sphinx-immaterial, myst-nb, sphinxcontrib-bibtex, numpydoc, sphinx-copybutton) in `docs/`; `nb_execution_mode='off'` renders stored notebook outputs
  - Exports `ClusterStratifiedKFold` from public API; adds `__version__`
  - Fixed typo + added `References` section in `area_of_applicability` docstring with `:cite:t:` for Meyer & Pebesma (2021)
  - Follow-ups: add `ippp.ipynb` to docs once committed; convert remaining docstrings to NumPy style with `:cite:` refs; add more `.bib` entries; consider `nb_execution_mode='cache'` once notebooks are stable

- [ ] **`metrics`** (not yet committed — `geovalidate.preprocessing` with `KNeighborsFeatures`, `RadiusNeighborsFeatures`)
  - Not committed — awaiting user review before committing
  - Adds `geovalidate.preprocessing` subpackage with two sklearn transformers:
    `KNeighborsFeatures` (k new columns per feature, one per neighbor rank) and
    `RadiusNeighborsFeatures` (one aggregated column per distance band per feature;
    supports annular rings `exclusive=True` or cumulative buffers `exclusive=False`)
  - Both accept GeoDataFrame (geometry auto-extracted) or explicit `geometry=` kwarg
  - Self-exclusion handled automatically when transforming training data (zero-distance check)
  - Both exported from top-level `geovalidate` namespace
  - Known follow-ups: add to `docs/source/api.rst`; write example notebook; add tests

## Cross-repo ideas to explore

- [ ] **esda: conditional permutation RNG refactor** — replace `vec_permutations` (Numba `np.random.choice` loop, forces joblib process-parallelism due to shared RNG state) with a pre-generated `(n_perm, max_card)` NumPy uniform matrix passed as read-only input to a Numba kernel that runs partial Fisher-Yates. Benefits: (a) NumPy PCG64 is faster than nopython-mode RNG; (b) `prange` becomes safe — no shared mutable state — eliminating joblib overhead and array copying to workers; (c) O(max_card) per permutation instead of O(n) (current `choice` shuffles all n). Discussed 2026-05-26.

- [ ] **geovalidate: conditional permutation test class** — a `ConditionalPermutationTest` that exposes esda's crand engine (or a reimplementation) as a sklearn-compatible metric for investigating model structure: given a fitted model, test whether spatially local residual patterns are consistent with spatial randomness, or whether specific covariates show spatially structured influence. Two-level API: slow general Python path for prototyping new statistics; fast compiled path for common cases. Natural home in `geovalidate.metrics` alongside `area_of_applicability`. Design note: the `stat_func(i, z, permuted_ids, weights_i, scaling) -> (n_perm,)` protocol from esda is already a reasonable plugin point.

## Planned metrics to implement

- [ ] **Local R²** — neighborhood-windowed R² for regression models. Two flavors to consider: (a) windowed evaluation of an externally fit model's predictions vs truth (cheap, useful as a CV diagnostic) and (b) per-point local-model R² (closer to GWR). Should mirror `LocalBootstrap`'s API for kernel / bandwidth / k inputs and reuse the `_utils.py` Graph plumbing. Pairs naturally with `area_of_applicability` (AOA = where the model *could* be applied; local R² = where it *actually scored* well).

- [ ] **Residual spatial autocorrelation** — Moran's I / Geary's C computed on CV residuals. The standard "did your spatial CV actually prevent leakage?" check. Global complement to local R²; together the two cover the post-prediction diagnostic story. `gearygram` handles the Geary's C correlogram part; still need a convenience wrapper that takes (y_true, y_pred, geometry) and computes residual autocorrelation.

- [x] **esda.map_comparison metrics** — wrap PySAL's `esda.map_comparison` (external/internal consistency, separability) for evaluating classifier and regionalization outputs. Useful for `ClusterStratifiedKFold`-style workflows and any classification task where you want to compare predicted vs true region partitions.

- [x] **Geosilhouettes from esda** — wrap user's own `esda.silhouettes` (boundary_silhouette, path_silhouette, silhouette_alist) with the same simpler interface as the rest of `geovalidate.metrics`. Spatial-aware silhouette scores for regionalization/cluster quality; pairs with the `esda.map_comparison` wrapper above as the two complementary regionalization-evaluation entry points (intra-region cohesion + cross-region agreement).

- [ ] **Feature-confounding diagnostic (sensemakr-like)** — given a predictive model and a focal feature, search for the feature (existing or constructable) that most reduces the focal feature's importance / partial-dependence signal when added to the model. Conceptually similar to Cinelli & Hazlett (2020) `sensemakr` for OLS, but generalized to arbitrary predictive models (use permutation importance or SHAP drop as the "destruction" metric). Precise definition + API needs design — likely a separate design pass before implementation.

- [ ] **Spatial ICE / local spillover maps** — spatial generalisation of Individual Conditional Expectation plots. Standard ICE varies feature k at site i and traces ŷ_i; this varies feature k at site i and traces ŷ_j for all j in the spatial neighbourhood, producing a *map* of propagated effects. Requires that the model encodes spatial coupling — spatial lags of covariates (W @ X), kernel-smoothed neighbourhood averages, or similar — so that x_i enters the feature vector of neighbouring j. Conceptually close to a spatial impulse response / multiplier (LeSage & Pace 2009) but model-agnostic (works for any sklearn-compatible model, not just spatial econometric models). Natural API: `SpatialICE(model, w, feature).compute(X, focal_site, grid)` returning a (n_grid, n) array of ΔŶ values that can be mapped spatially. Pairs with GeoSHAP-style attributions: SHAP decomposes a single prediction; spatial ICE shows the dynamic propagation of a counterfactual change across the network.

- [ ] **GeoShap-style attributions, simpler interface** — wrap an existing geo-SHAP implementation (most likely Li 2024 `geoshapley`, but confirm preferred backend) so the user gets location + feature attributions without dealing with the upstream's verbose API. Should accept the same `(X, y, model)` triple the other `geovalidate.metrics` entries use and return either a tidy DataFrame or a Bunch with `.intrinsic_location`, `.feature_main`, `.feature_geo_interaction`. Pairs well with local R² (attribution + diagnostic side by side).
