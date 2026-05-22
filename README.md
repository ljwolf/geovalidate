# geovalidate

Scikit-learn compatible spatial tools for geospatial model validation.

Motivated by the discussion in [geopandas/geopandas#3781](https://github.com/geopandas/geopandas/issues/3781),
this package provides four sampling strategies that follow the `sklearn.base.BaseEstimator`
API so they fit naturally into scikit-learn pipelines and cross-validation workflows.

## Samplers

Samplers are illustrated in the [`examples/usage.ipynb`](https://github.com/ljwolf/geovalidate/blob/main/examples/usage.ipynb) notebook. 

| Class | What it does |
|---|---|
| `PointSampler` | Uniform random points inside any Shapely geometry |
| `ConstantClassSampler` | Exactly *n* points from every class |
| `StratifiedClassSampler` | Fixed total allocated proportionally to a value column |
| `MultinomialSampler` | Multinomial allocation across label groups weighted by a per-geometry value |

Each sampler also accepts `quasi_random="sobol"`, `"halton"`, or `"r2"` to replace the default uniform RNG with a low-discrepancy sequence. Quasi-random sequences have better space-filling properties than points sampled at random, meaning that they tend to be less "clumpy" than randomly-sampled points. These options are only supported for some samplers.

## Installation

```bash
pip install -e ".[raster]"   # includes rasterio for raster-based samplers
```

## Quick start

```python
import geopandas, geodatasets
from geovalidate.samplers import (
  PointSampler,
  ConstantClassSampler,
  StratifiedClassSampler,
  MultinomialSampler
)

nybb = geopandas.read_file(geodatasets.get_path("nybb"))
```

### PointSampler — uniform random points inside a geometry

```python
# Sample from a single geometry
pts = PointSampler(n_samples=200, random_state=0).sample(nybb.geometry.iloc[0])

# Or from the union of a GeoSeries
pts = PointSampler(n_samples=500, random_state=0).sample(nybb.geometry)
```

### ConstantClassSampler — exactly n points per class

```python
# Pass geometry and a labels vector — no column names required
pts = ConstantClassSampler(n_per_class=80, random_state=0).sample(
    nybb.geometry, nybb["BoroName"]
)
pts.groupby("class_label").size()
```

### StratifiedClassSampler — total n allocated proportionally to weights

```python
# Weights control the allocation; counts are deterministic (greatest remainder is assigned the final sample)
pts = StratifiedClassSampler(n_samples=400, random_state=0).sample(
    nybb.geometry, nybb["BoroName"], nybb["Shape_Area"]
)

# Without weights, allocation is uniform across classes
pts = StratifiedClassSampler(n_samples=400, random_state=0).sample(
    nybb.geometry, nybb["BoroName"]
)
```

### MultinomialSampler — stochastic allocation from a multinomial draw

```python
# Class totals W_k = sum(weights where label == k)
# Counts ~ Multinomial(n_samples, W_k / ΣW_k), then points sampled uniformly per class
pts = MultinomialSampler(n_samples=500, random_state=0).sample(
    nybb.geometry, nybb["BoroName"], nybb["Shape_Area"]
)
```

### Raster input

Pass a rasterio `DatasetReader` as the geometry and read bands yourself:

```python
import rasterio

with rasterio.open("classes.tif") as ds:
    class_arr = ds.read(1)
    pts = ConstantClassSampler(n_per_class=60, random_state=0).sample(ds, class_arr)

with rasterio.open("classes.tif") as ds_c, rasterio.open("weights.tif") as ds_w:
    pts = MultinomialSampler(n_samples=500, random_state=0).sample(
        ds_c, ds_c.read(1), ds_w.read(1)
    )
```

### Quasi-random sequences

All samplers accept `quasi_random="sobol"`, `"halton"`, or `"r2"` for better spatial coverage:

```python
pts = PointSampler(n_samples=200, quasi_random="halton", random_state=0).sample(
    nybb.geometry.iloc[0]
)
```

### Crossvalidation tools

TBA: `LocalPermutation`, `LocalBootstrap`, `DispersionKFold`, `ClusterKFold`, `BallKFold`. 

### sklearn compatibility

Every sampler is a `BaseEstimator` subclass, so the standard sklearn utilities work:

```python
from sklearn.model_selection import GridSearchCV

# get / set hyperparameters
sampler = ConstantClassSampler(n_per_class=50, class_col="cls")
print(sampler.get_params())
sampler.set_params(n_per_class=200)
```