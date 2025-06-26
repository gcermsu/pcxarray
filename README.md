# `pcxarray`
A Python package for easy querying and access to Microsoft Planetary Computer data using xarray and geospatial tools.

## Features
- Query Microsoft Planetary Computer STAC API using shapely geometries
- Retrieve results as GeoDataFrames for easy inspection and filtering
- Download and preprocess raster data into xarray DataArrays
- Utilities for creating spatial grids and loading US Census TIGER shapefiles

## Installation

```bash
git clone https://github.com/DakotaHester/pcxarray.git
cd pcxarray
pip install -e .
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

# Create a grid over the state
grid_gdf = create_grid(
    ms_gdf.iloc[0].geometry,
    crs=ms_gdf.crs,
    cell_size=1000
)

# Query NAIP imagery for a grid cell
items_gdf = pc_query(
    collections='naip',
    geometry=grid_gdf.iloc[0].geometry,
    crs=grid_gdf.crs,
    datetime='2023'
)

# Download and load NAIP data as an xarray DataArray
imagery = prepare_data(
    geometry=grid_gdf.iloc[0].geometry,
    crs=grid_gdf.crs,
    items_gdf=items_gdf,
    target_resolution=1.0
)

# Or combine query and load in one step
imagery = query_and_prepare(
    collections='naip',
    geometry=grid_gdf.iloc[0].geometry,
    crs=grid_gdf.crs,
    datetime='2023',
    target_resolution=1.0
)
```
