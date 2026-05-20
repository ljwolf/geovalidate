"""
geovalidate — scikit-learn compatible spatial point samplers.

Four sampler classes extend the sklearn estimator API for geospatial
model-validation workflows:

    PointSampler      – uniform random points inside a Shapely geometry
    ConstantClassSampler        – fixed n points per class (GDF column or raster band)
    StratifiedClassSampler    – total n points allocated proportionally to class weight
    MultinomialSampler – IPPP where local density ∝ raster band / GDF column
"""

from .samplers import (
    PointSampler,
    ConstantClassSampler,
    StratifiedClassSampler,
    MultinomialSampler,
)
from .cv import SpatialKFold, GeoBootstrap, LocalPermutation

__all__ = [
    "PointSampler",
    "ConstantClassSampler",
    "StratifiedClassSampler",
    "MultinomialSampler",
    "SpatialKFold",
    "GeoBootstrap",
    "LocalPermutation",
]
