import numpy
import geopandas
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from scipy.sparse import diags
from scipy.spatial.distance import cdist

from ._utils import _get_coords, _to_point_gdf, KERNELS, LIBPYSAL_KERNEL_MAP


class LocalBootstrap(BaseEstimator):
    """Locally-weighted bootstrap — resampling with spatial (or temporal) proximity weighting.

    Generates *n_bootstraps* resampled datasets each of size n.  At every
    site i the observation placed there is drawn **with replacement** from all
    n observations; the probability is proportional to a kernel weight
    K(d_{ij} / bandwidth):

        p_{ij}  ∝  K(d_{ij} / bandwidth)

    so nearby observations are drawn more often.

    Works for **spatial** data (GeoDataFrame / (n, 2) coordinates) and for
    **time-series** data (pass a 1-D array of time indices; distance is then
    the absolute time lag).

    Passing a pre-built ``libpysal.graph.Graph`` via *graph* overrides
    *bandwidth* and *kernel*.

    Parameters
    ----------
    n_bootstraps : int, default 100
        Number of bootstrap resamples to generate.
    bandwidth : float or None
        Kernel bandwidth in the same units as the input distances.  Required
        unless *graph* is provided.
    kernel : str, default 'gaussian'
        One of ``'gaussian'``, ``'exponential'``, ``'bisquare'``,
        ``'triangular'``, ``'uniform'``, ``'parabolic'``.
    graph : libpysal.graph.Graph or None
        Pre-built spatial weights (must expose ``.sparse``).
        Overrides *bandwidth* / *kernel*.
    random_state : int, RandomState instance, or None

    Notes
    -----
    When input is a GeoDataFrame/GeoSeries and a supported kernel is given,
    a libpysal Graph is built internally — the weight matrix stays sparse.
    The dense O(n²) path is used only for raw array inputs or when kernel is
    ``'exponential'`` (not supported by libpysal).

    Explored initially in

    Statham, Thomas A. The Global Inconsistencies of Gridded Population Data
    at Different Spatial Scales. Diss. University of Bristol, 2024.

    Examples
    --------
    Spatial use (GeoDataFrame):

    >>> lb = LocalBootstrap(n_bootstraps=200, bandwidth=5_000,
    ...                     kernel='bisquare', random_state=0)
    >>> for indices in lb.sample(gdf):
    ...     boot = gdf.iloc[indices].copy()
    ...     boot.geometry = gdf.geometry.values   # restore original locations
    ...     model.fit(boot[features], y[indices])

    Time-series use (1-D array of time steps):

    >>> t = numpy.arange(len(df))
    >>> lb = LocalBootstrap(n_bootstraps=100, bandwidth=12, random_state=0)
    >>> for indices in lb.sample(t):
    ...     model.fit(X[indices], y[indices])
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        bandwidth: float | None = None,
        kernel: str = "gaussian",
        graph=None,
        random_state=None,
    ):
        self.n_bootstraps = n_bootstraps
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.graph = graph
        self.random_state = random_state

    def sample(self, X):
        """Yield locally-weighted bootstrap index arrays.

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Locations.  Pass a 1-D array of time indices for time-series data.

        Yields
        ------
        indices : ndarray of shape (n,)
            ``indices[i]`` is the source row drawn at position i.
            Apply with ``df.iloc[indices]`` (and restore original geometry /
            time index separately).
        """
        rng = check_random_state(self.random_state)

        if self.graph is not None:
            W_csr = self._sparse_weights_from_graph(self.graph)
            for _ in range(self.n_bootstraps):
                yield self._sample_csr(W_csr, rng)
            return

        if self.bandwidth is None:
            raise ValueError(
                "Specify either 'bandwidth' (kernel bandwidth) or "
                "a pre-built libpysal 'graph'."
            )

        if self.kernel not in KERNELS:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Choose from: {sorted(KERNELS)}."
            )

        # GeoDataFrame/GeoSeries → sparse path via libpysal kernel graph
        if isinstance(X, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
            from libpysal.graph import Graph
            point_gdf = _to_point_gdf(X)
            graph = Graph.build_kernel(
                point_gdf,
                bandwidth=self.bandwidth,
                kernel=LIBPYSAL_KERNEL_MAP[self.kernel],
            )
            W_csr = self._sparse_weights_from_graph(graph)
            for _ in range(self.n_bootstraps):
                yield self._sample_csr(W_csr, rng)
            return

        # Fallback: dense path for raw arrays / 1-D time-series
        coords = _get_coords(X)
        W = self._dense_weight_matrix(coords)
        for _ in range(self.n_bootstraps):
            yield self._sample_dense(W, rng)

    # ------------------------------------------------------------------
    # Weight matrix builders
    # ------------------------------------------------------------------

    def _dense_weight_matrix(self, coords: numpy.ndarray) -> numpy.ndarray:
        """Build a row-normalised (n, n) dense weight matrix."""
        if coords.shape[1] == 1:
            D = numpy.abs(coords - coords.T)
        else:
            D = cdist(coords, coords)
        W = KERNELS[self.kernel](D / self.bandwidth)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        return W / row_sums

    def _sparse_weights_from_graph(self, graph):
        """Return a row-normalised CSR sparse matrix — stays sparse throughout."""
        try:
            W = graph.sparse.tocsr().astype(float)
        except AttributeError:
            raise TypeError(
                "Expected a libpysal Graph with a '.sparse' attribute "
                "(scipy sparse matrix)."
            )
        row_sums = numpy.asarray(W.sum(axis=1)).ravel()
        row_sums = numpy.where(row_sums == 0, 1.0, row_sums)
        return diags(1.0 / row_sums) @ W

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_csr(W_csr, rng) -> numpy.ndarray:
        """Draw one index per row from a row-normalised CSR weight matrix.

        Uses ``numpy.searchsorted`` on each row's cumulative weights — O(k log k)
        per row where k is the row's nnz.  Never builds an (n, n) array.
        """
        n = W_csr.shape[0]
        u = rng.uniform(size=n)
        result = numpy.empty(n, dtype=numpy.intp)
        for i in range(n):
            start = int(W_csr.indptr[i])
            end = int(W_csr.indptr[i + 1])
            if start == end:  # isolated node — draw from self
                result[i] = i
                continue
            cumw = numpy.cumsum(W_csr.data[start:end])
            idx = numpy.searchsorted(cumw, u[i] * cumw[-1])
            result[i] = W_csr.indices[start + min(idx, end - start - 1)]
        return result

    @staticmethod
    def _sample_dense(W, rng) -> numpy.ndarray:
        """Draw one index per row from a row-normalised dense weight matrix."""
        n = W.shape[0]
        u = rng.uniform(size=n)
        cumW = W.cumsum(axis=1)
        return numpy.array(
            [numpy.searchsorted(cumW[i], u[i]) for i in range(n)], dtype=numpy.intp
        )
