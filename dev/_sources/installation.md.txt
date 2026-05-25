# Installation

## Stable release

```bash
pip install geovalidate
```

Include rasterio support for raster-based samplers:

```bash
pip install "geovalidate[raster]"
```

## Development version

```bash
pip install git+https://github.com/ljwolf/geovalidate.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/ljwolf/geovalidate.git
cd geovalidate
pip install -e ".[raster]"
```

## Dependencies

Core dependencies installed automatically:

- `geopandas >= 0.14`
- `scikit-learn >= 1.3`
- `shapely >= 2.0`
- `numpy >= 1.24`
- `scipy >= 1.7`
- `libpysal`

Optional:

- `rasterio >= 1.3` — raster-based intensity surfaces for `PoissonSampler`
