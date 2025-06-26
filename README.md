# `pcxarray`
[![PyPI version](https://img.shields.io/pypi/v/pcxarray.svg)](https://pypi.org/project/pcxarray/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Python package for easy querying and access to Microsoft Planetary Computer Data Catalogs using geopandas and xarray.

## Features
- Query Microsoft Planetary Computer STAC API using shapely geometries
- Retrieve results as GeoDataFrames for easy inspection and filtering
- Download and preprocess raster data into xarray DataArrays
- Utilities for creating spatial grids and loading US Census TIGER shapefiles

## Installation

`pcxarray` can be installed via pip.

```bash
python -m pip install pcxarray
```

Alternatively, you can install the development version directly from GitHub:

```bash
git clone https://github.com/gcermsu/pcxarray
cd pcxarray
python -m pip install -e ".[dev]"
```

## Usage

See `naip_demo.ipynb` for a complete example of querying NAIP imagery.

```python
from pcxarray import pc_query, prepare_data, query_and_prepare
from pcxarray.utils import create_grid, load_census_shapefile

# Load US state boundaries
states_gdf = load_census_shapefile(level="state")

# Select a state (e.g., Mississippi)
ms_gdf = states_gdf[states_gdf['STUSPS'] == 'MS']
ms_gdf = ms_gdf.to_crs(epsg=3814) # Reproject to a projected CRS (e.g., EPSG:3814 for Mississippi)

# Create a grid over the state
grid_gdf = create_grid(
    ms_gdf.iloc[0].geometry,
    crs=ms_gdf.crs,
    cell_size=1000 # each cell will be 1000 meters square (units depend on the CRS)
)
selected_geom = grid_gdf.iloc[10000].geometry # Select a single geometry for demonstration

# Query NAIP imagery for a grid cell
items_gdf = pc_query(
    collections='naip',
    geometry=selected_geom,
    crs=grid_gdf.crs,
    datetime='2023'
)

# Download and load NAIP data as an xarray DataArray - imagery is clipped to the 
# geometry of the given geometry, and a mosaic is created if the geometry spans 
# multiple indiviudual items.
imagery = prepare_data(
    geometry=selected_geom,
    crs=grid_gdf.crs,
    items_gdf=items_gdf,
    target_resolution=1.0
)

# Or combine query and load in one step
imagery = query_and_prepare(
    collections='naip',
    geometry=selected_geom,
    crs=grid_gdf.crs,
    datetime='2023',
    target_resolution=1.0
)
```
