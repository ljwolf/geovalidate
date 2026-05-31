"""Patch usage.ipynb in-place to reflect the current geovalidate API.

Only the cells whose API calls changed are touched; all user-added text,
styling, outputs, and colour choices are preserved exactly.
"""

import json

NOTEBOOK = "examples/usage.ipynb"

with open(NOTEBOOK) as f:
    nb = json.load(f)

by_id = {c.get("id", ""): c for c in nb["cells"]}


def _src(cell_id):
    s = by_id[cell_id]["source"]
    return "".join(s) if isinstance(s, list) else s


def code(cell_id, src):
    c = by_id[cell_id]
    c["source"] = src.strip()
    c["outputs"] = []
    c["execution_count"] = None


def md(cell_id, src):
    by_id[cell_id]["source"] = src.strip()


def patch(cell_id, old, new):
    by_id[cell_id]["source"] = _src(cell_id).replace(old, new)


# ── 1. Imports ────────────────────────────────────────────────────────────────
code("04126875", """
import geodatasets
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import colormaps

from geovalidate import (
    ConstantClassSampler,
    MultinomialSampler,
    PointSampler,
    StratifiedClassSampler,
)

%matplotlib inline
""")

# ── 2. Title table: IntensitySampler → MultinomialSampler ────────────────────
patch("7fd4a2d6",
    "| `IntensitySampler` | Inhomogeneous Poisson process — density ∝ a continuous field |",
    "| `MultinomialSampler` | Multinomial allocation across label groups weighted by a per-geometry value |",
)
patch("7fd4a2d6",
    "These options are only supported for some samplers. ",
    "These options are only supported for some samplers.",
)

# ── 3. GDF class sampler calls ────────────────────────────────────────────────
code("3ca01c5f", """
# ConstantClassSampler: exactly 80 pts from every borough
pts_const = ConstantClassSampler(n_per_class=80, random_state=0).sample(
    nybb.geometry, nybb["BoroName"]
)

# StratifiedClassSampler: 400 total, allocated ∝ Shape_Area
pts_strat = StratifiedClassSampler(n_samples=400, random_state=0).sample(
    nybb.geometry, nybb["BoroName"], nybb["Shape_Area"]
)

print("ConstantClassSampler — points per borough:")
print(pts_const.groupby("class_label").size().rename("n").to_string())
print()
print("StratifiedClassSampler — points per borough (∝ area):")
print(pts_strat.groupby("class_label").size().rename("n").to_string())
""")

# ── 4. Quasi-random ConstantClassSampler + StratifiedClassSampler ─────────────
code("580097ad", r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pts_const = ConstantClassSampler(n_per_class=80, random_state=85711, quasi_random="r2").sample(
    nybb.geometry, nybb["BoroName"]
)
pts_strat = StratifiedClassSampler(n_samples=400, random_state=85711, quasi_random="r2").sample(
    nybb.geometry, nybb["BoroName"], nybb["Shape_Area"]
)

for ax, pts, title in [
    (axes[0], pts_const, "ConstantClassSampler\n80 pts/borough — equal regardless of size"),
    (axes[1], pts_strat, "StratifiedClassSampler\n400 total — ∝ Shape_Area"),
]:
    for _, row in nybb.iterrows():
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax, color=BORO_COLORS[row.BoroName], alpha=0.2,
            edgecolor="#444", linewidth=0.8)
    for name, color in BORO_COLORS.items():
        sub = pts[pts["class_label"] == name]
        if len(sub):
            ax.scatter(
                sub.geometry.x,
                sub.geometry.y,
                s=12,
                color=color,
                zorder=3,
                edgecolors="white",
                linewidths=0.4
            )
    counts = pts.groupby("class_label").size()
    patches = [
        mpatches.Patch(
            color=BORO_COLORS[k],
            label=f"{k}  n={counts.get(k, 0)}"
        )
        for k in BORO_COLORS
    ]
    ax.legend(handles=patches, fontsize=8.5)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal"); ax.axis("off")

fig.tight_layout()
plt.show()
""")

# ── 5. MultinomialSampler markdown ────────────────────────────────────────────
md("6d4bc030", """
### `MultinomialSampler` on a GeoDataFrame

The `MultinomialSampler` works in two stages:

1. Sum *weights* within each *labels* group to get a per-class total weight
   `W_k`. Draw class sample counts jointly from
   `(n₁, n₂, …, nₖ) ~ Multinomial(n_samples, Wₖ / ΣWₖ)` — counts always
   sum to exactly `n_samples` but vary stochastically across runs.
2. Sample `nₖ` points uniformly from within the union of class `k`'s geometries.

This contrasts with `StratifiedClassSampler`, which allocates class counts
*deterministically* using the largest-remainder (Hamilton) method.  Here
we'll use `Shape_Area` as the per-borough weight, so larger boroughs attract
proportionally more sample points — but with random variation.
""")

# ── 6. MultinomialSampler GDF code ───────────────────────────────────────────
code("e9ad291d", """
pts_multi = MultinomialSampler(n_samples=500, random_state=0).sample(
    nybb.geometry, nybb["BoroName"], nybb["Shape_Area"]
)

print(f"n_samples=500, got {len(pts_multi)}")

fig, ax = plt.subplots(figsize=(6, 6))
nybb.plot(ax=ax, column="Shape_Area", cmap="YlOrRd", alpha=0.5,
          edgecolor="#444", linewidth=0.8, legend=False)
ax.scatter(pts_multi.geometry.x, pts_multi.geometry.y,
           s=5, c="#023e8a", alpha=0.65, zorder=3)
ax.set_title(f"MultinomialSampler — {len(pts_multi)} points  (∝ Shape_Area)")
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
""")

# ── 7. Raster section markdown ────────────────────────────────────────────────
patch("f19a9267",
    "| `intensity_mf` | `Shape_Area` of each borough's pixels | `IntensitySampler` |",
    "| `intensity_mf` | `Shape_Area` of each borough's pixels | `MultinomialSampler` |",
)

# ── 8. Raster ConstantClassSampler + StratifiedClassSampler ──────────────────
code("fcc30638", """
BORO_CODE = dict(zip(nybb.BoroCode, nybb.BoroName))

with class_mf.open() as ds:
    class_arr = ds.read(1)
    pts_const_r = ConstantClassSampler(n_per_class=60, random_state=0).sample(ds, class_arr)
    pts_strat_r = StratifiedClassSampler(n_samples=300, random_state=0).sample(ds, class_arr)

print("ConstantClassSampler — raster:")
print(pts_const_r.groupby("class_label").size().rename("n").to_string())
print()
print("StratifiedClassSampler — raster (∝ pixel count per BoroCode):")
print(pts_strat_r.groupby("class_label").size().rename("n").to_string())
""")

# ── 9. Raster MultinomialSampler (was IntensitySampler) ──────────────────────
code("56674b78", """
with class_mf.open() as ds_c, intensity_mf.open() as ds_i:
    pts_multi_r = MultinomialSampler(n_samples=500, random_state=0).sample(
        ds_c, ds_c.read(1), ds_i.read(1)
    )

print(f"n_samples=500, got {len(pts_multi_r)}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(arr_i, cmap="YlOrRd", origin="upper",
          extent=[*bounds[[0, 2, 1, 3]]], alpha=0.7)
ax.scatter(pts_multi_r.geometry.x, pts_multi_r.geometry.y,
           s=4, c="#023e8a", alpha=0.6, zorder=3)
ax.set_title(f"MultinomialSampler — {len(pts_multi_r)} points from raster")
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
""")

with open(NOTEBOOK, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Patched {NOTEBOOK}")
