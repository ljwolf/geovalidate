"""
geovalidate — scikit-learn compatible spatial point samplers and CV utilities.

Samplers:
    PointSampler                x uniform random points inside a Shapely geometry
    ConstantClassSampler        x fixed n points per class
    StratifiedClassSampler      x total n points allocated proportionally to weight
    MultinomialSampler          x stochastic class-count allocation via multinomial draw
    PoissonSampler              x true IPPP; intensity via callable, raster, or KDE, implementing spatstat

Cross-validation: [PLANNED]
    DispersionKFold   o spatially balanced k-fold (Hilbert curve interleaving)
    ClusterStratifiedKFold      o cluster-stratified k-fold (folds drawn proportionally from user-defined clusters)
    ExclusionBallFold o folds constructed such that no observation in any fold is within distance "r"
    LocalBootstrap    x locally-weighted bootstrap with replacement (spatial or 1-D)
    LocalPermutation  o spatially-constrained derangement without replacement
"""

from .samplers import (
    PointSampler,
    ConstantClassSampler,
    StratifiedClassSampler,
    MultinomialSampler,
    PoissonSampler,
)
from .cv import (
    ClusterStratifiedKFold,
    DispersionKFold,
    LocalBootstrap,
    LocalPermutation,
    correlogram_range,
    knn_range,
)


__all__ = [
    "PointSampler",
    "ConstantClassSampler",
    "StratifiedClassSampler",
    "MultinomialSampler",
    "PoissonSampler",
    "ClusterStratifiedKFold",
    "DispersionKFold",
    "LocalBootstrap",
    "LocalPermutation",
    "correlogram_range",
    "knn_range",
]
