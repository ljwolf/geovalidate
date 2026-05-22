import numpy
import geopandas
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from ._utils import _get_coords, _to_point_gdf, _idx_and_is_geo


class LocalPermutation(BaseEstimator):
    """Spatially-constrained permutation without replacement (derangement).

    Shuffles the rows of an n-site dataset so that each row moves **at most**
    *threshold* distance units from its origin (or only within the edges of a
    pre-built *graph*).  Every row appears exactly once -- it is a true
    permutation, analogous to :class:`LocalBootstrap` but **without**
    replacement.

    Optionally enforced as a **derangement**: no row may remain at its
    original site.

    Specifying *threshold* and *graph*
    -----------------------------------
    *threshold* alone
        Builds the adjacency from scratch: any pair of sites within
        *threshold* distance units may swap.
    *graph* alone
        Every directly-connected pair in the graph may swap (no distance
        filter on edge weights).
    *graph* **and** *threshold*
        Only directly-connected pairs whose stored edge weight is
        **≤ threshold** may swap.  No shortest-path computation is
        performed -- only direct (one-hop) edges are considered.

    Algorithm
    ---------
    1. Build a sparse boolean adjacency matrix A (feasible swap pairs).
    2. Find an initial feasible permutation via
       ``scipy.sparse.csgraph.min_weight_full_bipartite_matching`` on a
       sparse cost matrix whose entries are i.i.d. Uniform[0, 1] on
       feasible pairs -- giving a random feasible matching without
       materialising a dense (n, n) array.
    3. Mix with a **Markov chain**: propose swapping ``perm[i]`` and
       ``perm[j]``; accept iff both moves stay within A and neither
       creates a fixed point.  Adjacency lookups use a list of sets --
       O(1) average -- so no dense matrix is needed at any stage.
       Run *n_burn* proposed steps between each yielded permutation.

    Parameters
    ----------
    threshold : float or None
        Maximum allowed distance (or maximum edge weight when used with
        *graph*).  Required when *graph* is not provided.
    derangement : bool, default True
        If True, every value must move (no fixed points).
    n_permutations : int, default 99
        Number of permutations to generate.
    n_burn : int or None
        Proposed Markov-chain steps between each yielded permutation.
        Higher values give more independent samples at the cost of runtime.
        Defaults to ``10 * n``.
    graph : libpysal.graph.Graph or None
        Pre-built spatial weights (must expose ``.sparse``).  Edge weights
        are the values in the sparse matrix.  When *threshold* is also
        given, only edges with weight ≤ threshold are used.
    random_state : int, RandomState instance, or None

    Raises
    ------
    ValueError
        If no valid (de)rangement exists for the given constraints, or if
        neither *threshold* nor *graph* is supplied.

    Examples
    --------
    Distance threshold only:

    >>> lp = LocalPermutation(threshold=50_000, random_state=0)
    >>> for perm in lp.sample(gdf):
    ...     model.fit(X[perm], y[perm])

    Graph with an additional weight filter (only edges ≤ 10 km):

    >>> lp = LocalPermutation(graph=W, threshold=10_000, random_state=0)
    >>> for perm in lp.sample(gdf):
    ...     model.fit(X[perm], y[perm])
    """

    def __init__(
        self,
        threshold: float | None = None,
        derangement: bool = True,
        n_permutations: int = 99,
        n_burn: int | None = None,
        graph=None,
        random_state=None,
    ):
        self.threshold = threshold
        self.derangement = derangement
        self.n_permutations = n_permutations
        self.n_burn = n_burn
        self.graph = graph
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, X):
        """Yield constrained permutation index arrays.

        ``perm[i] = j`` means site i receives the row from site j.
        Apply as ``y[perm]`` to permute a target vector, or
        ``X[perm]`` / ``gdf.iloc[perm]`` to permute all features jointly
        (keep the original geometry / time index separately).

        Parameters
        ----------
        X : GeoDataFrame | GeoSeries | (n, 2) ndarray | (n,) or (n, 1) ndarray
            Locations.  When *graph* is provided coordinates are used only
            to determine n (pass any array of the right length if
            coordinates are not meaningful).

        Yields
        ------
        perm : ndarray of shape (n,)
            ``perm[i]`` is the label of the row assigned to position i,
            using the input's index when X is a GeoDataFrame/GeoSeries, or
            integer positions for raw arrays.
        """
        rng = check_random_state(self.random_state)
        idx, is_geo = _idx_and_is_geo(X)

        if self.graph is not None:
            n = self._n_from_graph(self.graph)
            adj_csr, adj_sets = self._adj_from_graph(self.graph, n)
        else:
            if self.threshold is None:
                raise ValueError(
                    "Specify either 'threshold' (distance cutoff), "
                    "a pre-built libpysal 'graph', or both."
                )
            if is_geo:
                from libpysal.graph import Graph

                point_gdf = _to_point_gdf(X)
                graph = Graph.build_distance_band(point_gdf, threshold=self.threshold)
                n = len(point_gdf)
                adj_csr, adj_sets = self._adj_from_graph(graph, n)
            else:
                coords = _get_coords(X)
                n = len(coords)
                adj_csr, adj_sets = self._build_adj(coords, n)

        n_burn = self.n_burn if self.n_burn is not None else 10 * n
        perm = self._initial_permutation(adj_csr, n, rng)

        for _ in range(self.n_permutations):
            perm = self._markov_mix(perm, adj_sets, n_burn, rng)
            positions = perm.copy()
            yield idx[positions] if idx is not None else positions

    # ------------------------------------------------------------------
    # Adjacency builders -- both return (adj_csr, adj_sets)
    # ------------------------------------------------------------------

    def _build_adj(self, coords: numpy.ndarray, n: int):
        """Sparse adjacency from a coordinate distance threshold."""
        tree = cKDTree(coords)
        pairs = tree.query_pairs(self.threshold)  # set of (i,j), i < j

        rows, cols = [], []
        for i, j in pairs:
            rows += [i, j]
            cols += [j, i]

        if not self.derangement:
            rows += list(range(n))
            cols += list(range(n))

        data = numpy.ones(len(rows), dtype=bool)
        adj_csr = csr_matrix((data, (rows, cols)), shape=(n, n))
        adj_csr.sum_duplicates()

        adj_sets = self._sets_from_pairs(rows, cols, n)
        return adj_csr, adj_sets

    def _adj_from_graph(self, graph, n: int):
        """Sparse adjacency from graph edges, optionally filtered by threshold."""
        try:
            W = graph.sparse.tocsr().astype(float)
        except AttributeError:
            raise TypeError(
                "Expected a libpysal Graph with a '.sparse' attribute "
                "(scipy sparse matrix)."
            )

        # Convert to COO to filter entries without going dense
        W_coo = W.tocoo()
        mask = W_coo.data != 0

        if self.threshold is not None:
            mask &= W_coo.data <= self.threshold

        row = W_coo.row[mask]
        col = W_coo.col[mask]

        # Remove self-loops (derangement: diagonal stays absent)
        off_diag = row != col
        row, col = row[off_diag], col[off_diag]

        if not self.derangement:
            diag = numpy.arange(n)
            row = numpy.concatenate([row, diag])
            col = numpy.concatenate([col, diag])

        data = numpy.ones(len(row), dtype=bool)
        adj_csr = csr_matrix((data, (row, col)), shape=(n, n))
        adj_csr.sum_duplicates()

        adj_sets = self._sets_from_pairs(row, col, n)
        return adj_csr, adj_sets

    @staticmethod
    def _sets_from_pairs(rows, cols, n) -> list:
        """Build a list-of-sets adjacency for O(1) Markov-chain lookups."""
        adj_sets: list[set] = [set() for _ in range(n)]
        for i, j in zip(rows, cols):
            adj_sets[i].add(j)
        return adj_sets

    @staticmethod
    def _n_from_graph(graph) -> int:
        try:
            return graph.n
        except AttributeError:
            raise TypeError("Expected a libpysal Graph with a '.sparse' attribute.")

    # ------------------------------------------------------------------
    # Initial permutation -- sparse assignment
    # ------------------------------------------------------------------

    def _initial_permutation(self, adj_csr, n: int, rng) -> numpy.ndarray:
        """Find a random feasible matching via min_weight_full_bipartite_matching.

        Assigns i.i.d. Uniform[0,1] weights to feasible pairs so the
        minimum-weight solution is effectively a random feasible matching.
        The sparse cost matrix is never expanded to a dense (n, n) array.
        """
        rows, cols = adj_csr.nonzero()
        weights = rng.uniform(0.0, 1.0, len(rows)).astype(float)
        cost_csr = csr_matrix((weights, (rows, cols)), shape=(n, n))

        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(cost_csr)
        except ValueError:
            kind = "derangement" if self.derangement else "permutation"
            src = (
                "the supplied graph"
                if self.graph is not None
                else f"threshold={self.threshold}"
            )
            if self.graph is not None and self.threshold is not None:
                src = f"the supplied graph filtered to edges ≤ {self.threshold}"
            raise ValueError(
                f"No valid constrained {kind} exists within {src}.  "
                "Increase the threshold, use a denser graph, or set "
                "derangement=False."
            )

        perm = numpy.empty(n, dtype=int)
        perm[row_ind] = col_ind
        return perm

    # ------------------------------------------------------------------
    # Markov chain -- set-based adjacency lookups
    # ------------------------------------------------------------------

    def _markov_mix(
        self,
        perm: numpy.ndarray,
        adj_sets: list,
        n_steps: int,
        rng,
    ) -> numpy.ndarray:
        """Random-transposition Markov chain using O(1) set lookups.

        Proposes swapping perm[i] <-> perm[j] and accepts iff:
          - vj is a valid destination for site i  (vj in adj_sets[i])
          - vi is a valid destination for site j  (vi in adj_sets[j])
          - neither creates a fixed point (derangement constraint)
        """
        n = len(perm)
        perm = perm.copy()

        for _ in range(n_steps):
            i, j = rng.choice(n, size=2, replace=False)
            vi, vj = perm[i], perm[j]

            if vj not in adj_sets[i] or vi not in adj_sets[j]:
                continue
            if self.derangement and (vj == i or vi == j):
                continue

            perm[i], perm[j] = vj, vi

        return perm
