import pytest
import geopandas as gpd
from shapely.geometry import box


@pytest.fixture
def square():
    """10×10 unit square."""
    return box(0, 0, 10, 10)


@pytest.fixture
def class_gdf():
    """Three polygons, two classes.  Class A sum(value)=30, class B sum=30."""
    geoms = [box(0, 0, 5, 5), box(5, 0, 10, 5), box(0, 5, 10, 10)]
    return gpd.GeoDataFrame(
        {"class": ["A", "A", "B"], "value": [10.0, 20.0, 30.0], "geometry": geoms}
    )


@pytest.fixture
def multinomial_gdf():
    """Four equal-area boxes in two classes with varied weights.

    Class A: weights [1, 9]  → total 10
    Class B: weights [4, 6]  → total 10
    Equal class totals → 50/50 expected split.
    """
    geoms = [box(0, 0, 5, 5), box(5, 0, 10, 5), box(0, 5, 5, 10), box(5, 5, 10, 10)]
    return gpd.GeoDataFrame({
        "class":  ["A", "A", "B", "B"],
        "weight": [1.0, 9.0, 4.0, 6.0],
        "geometry": geoms,
    })
