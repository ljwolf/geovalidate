"""
geovalidate — scikit-learn compatible spatial point samplers and CV utilities.

Samplers:
    PointSampler                x uniform random points inside a Shapely geometry
    ConstantClassSampler        x fixed n points per class
    StratifiedClassSampler      x total n points allocated proportionally to weight
    MultinomialSampler          x stochastic class-count allocation via multinomial draw
    PoissonSampler              x true IPPP; intensity via callable, raster, or KDE, implementing spatstat

Cross-validation: [PLANNED]
    HilbertKFold      o spatially balanced k-fold (Hilbert curve interleaving)
    BallKFold         o spatially exclusive k-fold (no two samples in a fold within radius r)
    ClusterKFold      o spatially concentrated k-fold (user-specified location clusterer)
    LocalBootstrap    x locally-weighted bootstrap with replacement (spatial or 1-D)
    LocalPermutation  o spatially-constrained derangement without replacement

Preprocessing:
    KNeighborsFeatures      x k nearest-neighbor feature columns (nn1 ... nnk per feature)
    RadiusNeighborsFeatures x distance-band aggregated feature columns (annular or cumulative)

Metrics:
    area_of_applicability  x Meyer & Pebesma 2021 Area of Applicability
    gearygram              x Geary's C correlogram (bandwidth / kNN / nonparametric)
"""

from .samplers import (
    PointSampler,
    ConstantClassSampler,
    StratifiedClassSampler,
    MultinomialSampler,
    PoissonSampler,
)
from .cv import (
    BallKFold,
    LeaveBallOut,
    ClusterStratifiedKFold,
    DGGridStratifiedKFold,
    HilbertKFold,
    LocalBootstrap,
    LocalPermutation,
    correlogram_range,
    knn_range,
)
from .metrics import area_of_applicability, gearygram
from .preprocessing import KNeighborsFeatures, RadiusNeighborsFeatures

__version__ = "0.1.0"

__all__ = [
    "PointSampler",
    "ConstantClassSampler",
    "StratifiedClassSampler",
    "MultinomialSampler",
    "PoissonSampler",
    "BallKFold",
    "LeaveBallOut",
    "ClusterStratifiedKFold",
    "DGGridStratifiedKFold",
    "HilbertKFold",
    "LocalBootstrap",
    "LocalPermutation",
    "correlogram_range",
    "knn_range",
    "area_of_applicability",
    "gearygram",
    "KNeighborsFeatures",
    "RadiusNeighborsFeatures",
]
