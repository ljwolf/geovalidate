# geovalidate

Scikit-learn compatible spatial point samplers for geospatial model validation.

Motivated by the discussion in [geopandas/geopandas#3781](https://github.com/geopandas/geopandas/issues/3781),
this package provides four sampling strategies that follow the `sklearn.base.BaseEstimator`
API so they fit naturally into scikit-learn pipelines and cross-validation workflows.

## Samplers

| Class | Description |
|---|---|
| `PointSampler` | Uniform random points inside any Shapely geometry |
| `ConstantClassSampler` | Exactly *n* points per class (GeoDataFrame column or raster band) |
| `StratifiedClassSampler` | Fixed total *n*, allocated across classes proportionally to class weight |
| `IntensitySampler` | IPPP — local density proportional to a raster band or GDF column |

All samplers accept either a **GeoDataFrame** (with an appropriate column) or a
**rasterio-compatible raster** (with an appropriate band).

## Installation

```bash
pip install -e ".[raster]"   # includes rasterio for raster-based samplers
```

## Quick start

```python
from shapely.geometry import box
import geopandas as gpd
from geovalidate import (
    PointSampler,
    ConstantClassSampler,
    StratifiedClassSampler,
    IntensitySampler,
)

# 1 — uniform points inside a geometry
pts = PointSampler(n_samples=500, random_state=0).sample(box(0, 0, 1, 1))

# 2 — balanced: 100 points per land-cover class
gdf = gpd.read_file("landcover.gpkg")
pts = ConstantClassSampler(n_per_class=100, class_col="lc_class").sample(gdf)

# 3 — proportional: 1000 points total, weighted by per-class area sum
pts = StratifiedClassSampler(
    n_samples=1000, class_col="lc_class", value_col="area_ha"
).sample(gdf)

# 4 — IPPP from a raster intensity band
import rasterio
with rasterio.open("ndvi.tif") as ds:
    pts = IntensitySampler(n_samples=2000, band=1).sample(ds)
```

## sklearn compatibility

Every sampler is a `BaseEstimator` subclass, so the standard sklearn utilities work:

```python
from sklearn.model_selection import GridSearchCV

# get / set hyperparameters
sampler = ConstantClassSampler(n_per_class=50, class_col="cls")
print(sampler.get_params())
sampler.set_params(n_per_class=200)
```

## Design notes

- **`ConstantClassSampler`** mirrors `StratifiedKFold`: the class distribution in the
  output is perfectly balanced regardless of how skewed the source data is.
- **`StratifiedClassSampler`** uses the largest-remainder (Hamilton) method for
  integer allocation, guaranteeing `sum(per-class counts) == n_samples` exactly.
- **`IntensitySampler`** draws Poisson-distributed counts per pixel / geometry,
  which is the statistically correct IPPP behaviour.  The returned count therefore varies
  around *n_samples*.
- Rasterio is an optional dependency.  Install `geovalidate[raster]` if you need
  raster-based sampling.
