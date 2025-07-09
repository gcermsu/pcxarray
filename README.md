# pcxarray
[![PyPI version](https://img.shields.io/pypi/v/pcxarray.svg)](https://pypi.org/project/pcxarray/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Planetary Computer + Xarray**: A Python package for seamless querying, downloading, and processing of Microsoft Planetary Computer raster data using GeoPandas and Xarray.

## Overview

`pcxarray` bridges the gap between Microsoft's Planetary Computer STAC API and modern Python geospatial workflows. It enables you to query satellite imagery using simple geometries and automatically load the results as analysis-ready xarray DataArrays with proper spatial reference handling, mosaicking, and preprocessing.

### Key Concepts

- **Geometry-based queries**: Use any shapely geometry to define your area of interest
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

The typical `pcxarray` workflow follows three main steps:

### 1. Define Your Area of Interest

```python
from shapely.geometry import Polygon
import geopandas as gpd

# Create a geometry (CRS is important - results will match this CRS)
geom = Polygon([...])  # Your area of interest
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

## Working with Large Datasets

`pcxarray` is designed for Dask's lazy execution model, making it efficient for large datasets:

```python
from distributed import Client

# Start Dask client for parallel processing
client = Client(n_workers=4, memory_limit="4GB")

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

Explore these comprehensive examples in the [`examples/`](examples/) directory:

- **[`naip_demo.ipynb`](naip_demo.ipynb)**: NAIP imagery processing with grid creation
- **[`hls_timeseries.ipynb`](examples/hls_timeseries.ipynb)**: Water quality monitoring with HLS data
- **[`sentinel2_timeseries.ipynb`](examples/sentinel2_timeseries.ipynb)**: Vegetation monitoring with Sentinel-2
- **[`landsat_timeseries.ipynb`](examples/landsat_timeseries.ipynb)**: Long-term change analysis with Landsat
- **[`gnatsgo.ipynb`](examples/gnatsgo.ipynb)**: Soil productivity mapping

## Known Issues

- Inconsistent chunking behavior when passing `bands` dimension in `chunks` dict in `prepare_data`
- Some collections may have different metadata schemas causing issues. If an issue is encountered, please open an issue on GitHub.
