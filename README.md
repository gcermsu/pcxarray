# pcxarray 

A Python package for seamless querying, downloading, and processing of Microsoft Planetary Computer raster data using GeoPandas and Xarray.

[![PyPI version](https://img.shields.io/pypi/v/pcxarray.svg)](https://pypi.org/project/pcxarray/)
[![Downloads](https://pepy.tech/badge/pcxarray)](https://pepy.tech/project/pcxarray)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://pcxarray.readthedocs.io/en/stable/license.html)
[![Documentation Status](https://readthedocs.org/projects/pcxarray/badge/?version=stable)](https://pcxarray.readthedocs.io/en/stable/?badge=stable)
![example workflow](https://github.com/gcermsu/pcxarray/actions/workflows/pypi-release.yml/badge.svg)
[![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/hls_timeseries.ipynb)

## Overview

`pcxarray` (Planetary Computer + xarray) bridges the gap between Microsoft's Planetary Computer Data Catalog and modern Python geospatial workflows. It enables querying satellite imagery using simple geometries and automatically loads the results as analysis-ready xarray DataArrays with proper spatial reference handling, mosaicking, and preprocessing. This package is designed to work seamlessly with Dask for lazy execution and distributed processing, making it ideal for large-scale geospatial data analysis.

### Key Concepts

- **Geometry-based queries**: Use any shapely geometry to define areas of interest
- **Automatic spatial processing**: Handle reprojection, resampling, and mosaicking transparently  
- **Dask integration**: Lazy loading and parallel processing for large datasets
- **Analysis-ready data**: Get properly georeferenced xarray DataArrays ready for analysis

## Features

- **Query Microsoft Planetary Computer STAC API** using shapely geometries
- **Retrieve results as GeoDataFrames** for inspection, filtering, and spatial analysis
- **Download and mosaic raster data** into xarray DataArrays with reprojection and resampling
- **Create timeseries datasets** from multiple satellite acquisitions
- **Utilities for spatial analysis**: grid creation and US Census TIGER shapefiles
- **Simple caching** of expensive or repeated downloads
- **Designed for integration** with Dask, Jupyter, and modern geospatial Python workflows

## Installation

Install from PyPI:

```bash
python -m pip install pcxarray
```

Or install the development version from GitHub:

```bash
git clone https://github.com/gcermsu/pcxarray
cd pcxarray
python -m pip install -e ".[dev]"
```

## Core Workflow

For a comprehensive quickstart guide, see the [HLS time series example](examples/hls_timeseries.ipynb) [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/hls_timeseries.ipynb)

The typical `pcxarray` workflow follows three main steps:

### 1. Define Area of Interest

```python
from shapely.geometry import Polygon
import geopandas as gpd

# Create a geometry (CRS is important - results will match this CRS)
geom = Polygon([...])  # Area of interest
gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:32616")  # Project to appropriate UTM zone
roi_geom = gdf.geometry.values[0]
```

### 2. Query Planetary Computer

```python
from pcxarray import pc_query

# Query for satellite data
items_gdf = pc_query(
    collections='sentinel-2-l2a',  # Collection ID
    geometry=roi_geom,
    datetime='2024-01-01/2024-12-31',  # RFC 3339 datetime
    crs=gdf.crs
)
print(f"Found {len(items_gdf)} items")
```

### 3. Load and Process Data

```python
from pcxarray import prepare_data

# Load as xarray DataArray
imagery = prepare_data(
    items_gdf=items_gdf,
    geometry=roi_geom,
    crs=gdf.crs,
    bands=['B04', 'B03', 'B02'],  # Red, Green, Blue
    target_resolution=10.0,  # meters
    merge_method='mean'
)

# Visualize
(imagery / 3000).plot.imshow()
```
## Quick Examples

### NAIP Imagery
```python
from pcxarray import query_and_prepare
from pcxarray.utils import create_grid, load_census_shapefile

# Load state boundaries and create processing grid
states_gdf = load_census_shapefile(level="state")
ms_gdf = states_gdf[states_gdf['STUSPS'] == 'MS'].to_crs(3814)

# Create 1km grid and select a cell
grid_gdf = create_grid(ms_gdf.iloc[0].geometry, crs=ms_gdf.crs, cell_size=1000)
selected_geom = grid_gdf.iloc[10000].geometry

# Query and load NAIP imagery
imagery = query_and_prepare(
    collections='naip',
    geometry=selected_geom,
    crs=ms_gdf.crs,
    datetime='2023',
    target_resolution=1.0,
    bands=[4, 1, 2]  # NIR, Red, Green
)
```

### Satellite Timeseries Analysis
```python
from pcxarray import prepare_timeseries
import xarray as xr

# Query multiple years of Landsat data
items_gdf = pc_query(
    collections="landsat-c2-l2",
    geometry=roi_geom,
    datetime="2020/2024", 
    crs=utm_crs,
    # query={"eo:cloud_cover": {"lt": 5}}  # Optional cloud cover filter
)

# Create timeseries DataArray
timeseries = prepare_timeseries(
    items_gdf=items_gdf,
    geometry=roi_geom,
    crs=utm_crs,
    bands=["green", "nir08"],
    chunks={"time": 16, "x": 2048, "y": 2048}
)

# Convert from DN to reflectance
timeseries = (timeseries * 0.0000275) - 0.2

# Calculate NDVI timeseries
ndvi = (timeseries.sel(band="nir08") - timeseries.sel(band="green")) / \
       (timeseries.sel(band="nir08") + timeseries.sel(band="green"))

# Compute monthly means
monthly_ndvi = ndvi.resample(time="1M").mean().persist() # use lazy execution
```

For more complete examples, see the [`examples/`](examples/) directory.

## Working with Large Datasets

`pcxarray` is designed for Dask's lazy execution model, making it efficient for large datasets:

```python
from distributed import Client

# Start Dask client for parallel processing
client = Client(processes=True)

# Prepare timeseries (creates computation graph, doesn't load data)
da = prepare_timeseries(
    items_gdf=large_items_gdf,
    geometry=roi_geom, 
    crs=target_crs,
    bands=["B04", "B08"],
    chunks={"time": 32, "x": 2048, "y": 2048}
)

# Process data (computation happens here)
result = da.resample(time="1M").mean().compute()
```

## Supported Collections

`pcxarray` works with Microsoft Planetary Computer collections that provide data in Cloud Optimized GeoTIFF (COG) format and are accessible via the STAC API. The collections are identified by their unique IDs, which can be used in queries to retrieve data. Popular examples include:

- **Landsat**: `landsat-c2-l2` (Landsat Collection 2 Level-2)
- **Sentinel-2**: `sentinel-2-l2a` (Sentinel-2 Level-2A) 
- **NAIP**: `naip` (National Agriculture Imagery Program)
- **HLS**: `hls2-l30`, `hls2-s30` (Harmonized Landsat Sentinel-2)
- **Soil Data**: `gnatsgo-rasters` (Gridded National Soil Survey)

Use `get_pc_collections()` to discover available collections. Note that not all collections are compatible with `pcxarray`, such those that do not provide COGs (such as Sentinel-3/5p collections) or those that are not cataloged in the Planetary Computer STAC API (such as NLCD). Consult the [Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/catalog) for a complete list of available datasets.

## Complete Examples

Explore these comprehensive examples in the [`examples/`](examples/) directory of the repository:

- **[`hls_timeseries.ipynb`](examples/hls_timeseries.ipynb)**: Water quality monitoring with HLS data [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/hls_timeseries.ipynb)
- **[`naip.ipynb`](examples/naip.ipynb)**: NAIP imagery processing with grid creation [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/naip.ipynb)
- **[`sentinel2_timeseries.ipynb`](examples/sentinel2_timeseries.ipynb)**: Vegetation monitoring with Sentinel-2 [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/sentinel2_timeseries.ipynb)
- **[`landsat_timeseries.ipynb`](examples/landsat_timeseries.ipynb)**: Long-term change analysis with Landsat [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/landsat_timeseries.ipynb)
- **[`gnatsgo.ipynb`](examples/gnatsgo.ipynb)**: Soil productivity mapping [![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter)](https://nbviewer.org/github/gcermsu/pcxarray/blob/main/examples/gnatsgo.ipynb)

## API Reference

Documentation can be found at [pcxarray.readthedocs.io](https://pcxarray.readthedocs.io/en/stable/).

## Known Issues

- Chunking along `band` or `time` dimension when preparing timeseries datasets can trigger rechunks, which may be undesirable.
- Some collections may have different metadata schemas causing issues. If an issue is encountered, please open an issue on GitHub.
- When using Dask distributed scheduler, `open_rasterio` tasks may get stuck and prevent the computation graph from fully executing. Initializing the Dask client with `processes=True` seems to resolve this.

## Acknowledgements

This package is developed and maintained by the [GCERLab](https://www.gcerlab.com/) group at Mississippi State University. We welcome contributions and feedback from the community. If you find any issues or have feature requests, please open an issue on GitHub. Pull requests are also welcome!