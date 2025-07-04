# pcxarray
[![PyPI version](https://img.shields.io/pypi/v/pcxarray.svg)](https://pypi.org/project/pcxarray/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Python package for querying, downloading, and processing Microsoft Planetary Computer raster data using GeoPandas and Xarray.

## Features
- Query Microsoft Planetary Computer STAC API using shapely geometries
- Retrieve results as GeoDataFrames for inspection, filtering, and spatial analysis
- Download and mosaic raster data into xarray DataArrays with reprojection and resampling
- Utilities for creating spatial grids and loading US Census TIGER shapefiles
- Simple caching of expensive or repeated downloads
- Designed for integration with Dask, Jupyter, and modern geospatial Python workflows

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

## Quickstart Example

See [`naip_demo.ipynb`](naip_demo.ipynb) for a complete example of querying and visualizing NAIP imagery.

```python
from pcxarray import pc_query, prepare_data, query_and_prepare
from pcxarray.utils import create_grid, load_census_shapefile

# Load US state boundaries
states_gdf = load_census_shapefile(level="state")

# Select a state (e.g., Mississippi)
ms_gdf = states_gdf[states_gdf['STUSPS'] == 'MS']
ms_gdf = ms_gdf.to_crs(3814)  # Project to Mississippi Transverse Mercator

# Create a 1km grid over the state
grid_gdf = create_grid(
    ms_gdf.iloc[0].geometry,
    crs=ms_gdf.crs,
    cell_size=1000
)
selected_geom = grid_gdf.iloc[10000].geometry  # Select a single grid cell

# Query NAIP imagery for the selected cell
items_gdf = pc_query(
    collections='naip',
    geometry=selected_geom,
    crs=grid_gdf.crs,
    datetime='2023'
)

# Download and load NAIP data as an xarray DataArray
imagery = prepare_data(
    items_gdf=items_gdf,
    geometry=selected_geom,
    crs=grid_gdf.crs,
    target_resolution=1.0,
    bands=[4, 1, 2],  # NIR, Red, Green
    merge_method='first'
)

# Or combine query and load in one step
imagery = query_and_prepare(
    collections='naip',
    geometry=selected_geom,
    crs=grid_gdf.crs,
    datetime='2023',
    target_resolution=1.0,
    bands=[4, 1, 2]
)

# Visualize (in Jupyter)
(imagery / 255.0).plot.imshow()
```

## Core Functions
- `pc_query`: Query the Planetary Computer STAC API and return results as a GeoDataFrame
- `prepare_data`: Download, mosaic, and preprocess raster data for a given geometry
- `query_and_prepare`: Convenience function to query and load data in one step
- `create_grid`: Generate a regular grid of polygons over a region
- `load_census_shapefile`: Download and load US Census TIGER shapefiles (state, county, ZCTA)

## More Examples
See [`examples/`](/examples/) for full deomstraations workflows, including grid creation, querying, and visualization.

## Known Issues
- Inconsistent chunking behavior when passing `bands` dimension in `chunks` dict in `prepare_data`. 
