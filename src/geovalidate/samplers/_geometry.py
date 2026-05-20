import geopandas as gpd
import shapely
from sklearn.utils import check_random_state

from ._base import BasePointSampler
from ._utils import _sample_geometry


class PointSampler(BasePointSampler):
    """Sample points uniformly at random inside a Shapely geometry.

    Mimics the sklearn model-selection estimator API: hyperparameters are set
    in ``__init__`` and :meth:`sample` acts as the primary callable.

    Parameters
    ----------
    n_samples : int, default 100
        Number of points to generate.
    random_state : int, RandomState instance, or None, default None
        Seed / random state passed to ``sklearn.utils.check_random_state``.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> from geovalidate import PointSampler
    >>> pts = PointSampler(n_samples=200, random_state=0).sample(box(0, 0, 1, 1))
    >>> len(pts)
    200
    """

    def __init__(self, n_samples: int = 100, quasi_random: str | None = None,
                 random_state=None):
        self.n_samples = n_samples
        self.quasi_random = quasi_random
        self.random_state = random_state

    def sample(
        self,
        geometry,
        n_samples: int | None = None,
        crs=None,
    ) -> gpd.GeoDataFrame:
        """Sample *n_samples* points uniformly inside *geometry*.

        Parameters
        ----------
        geometry : shapely.Geometry | gpd.GeoSeries | gpd.GeoDataFrame
            Region to sample from.  A GeoSeries / GeoDataFrame is dissolved
            into a single union before sampling.
        n_samples : int, optional
            Overrides ``self.n_samples`` for this call.
        crs : CRS-like, optional
            CRS to attach to the returned GeoDataFrame.  Inferred automatically
            when *geometry* is a GeoSeries / GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            Single-column ``geometry`` GeoDataFrame of sampled Points.
        """
        rng = check_random_state(self.random_state)
        n = n_samples if n_samples is not None else self.n_samples

        if isinstance(geometry, gpd.GeoDataFrame):
            crs = crs or geometry.crs
            geometry = geometry.geometry.union_all()
        elif isinstance(geometry, gpd.GeoSeries):
            crs = crs or geometry.crs
            geometry = geometry.union_all()

        pts = _sample_geometry(geometry, n, rng, self.quasi_random)
        return gpd.GeoDataFrame(geometry=pts, crs=crs)
