import os
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import geopandas as gpd
import numpy as np
from pyproj import CRS
import requests
from shapely.geometry import Polygon
from shapely import prepare
from tqdm import tqdm
from typing import Literal, Union
from .cache import cache

def create_grid(
    polygon: Polygon,
    crs: Union[CRS, str, int] = 4326,
    cell_size: int = 1000,
    enable_progress_bar: bool = False,
    clip_to_polygon: bool = True,
) -> gpd.GeoDataFrame:
    """
    Generate a regular grid of square polygons covering a given input polygon.

    The grid is aligned to the bounding box of the input polygon and consists of 
    square cells of the specified size. Optionally, only grid cells that intersect 
    the input polygon are retained, and the cells can be clipped to the polygon 
    boundary.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon to cover with a grid.
    crs : pyproj.CRS, str or int, default=4326
        The coordinate reference system for the output GeoDataFrame.
    cell_size : int, default=1000
        The size of each grid cell along each side in the units of the CRS.
    enable_progress_bar : bool, default=False
        Whether to display progress bars during grid creation and filtering.
    clip_to_polygon : bool, default=True
        If True, grid cells will be clipped to the input polygon boundary.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the grid polygons, optionally clipped to the input polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)
    
    # Create grid of polygons using numpy broadcasting
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    grid_polygons = [
        Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
        for x, y in tqdm(zip(x_grid.ravel(), y_grid.ravel()), desc="Creating grid polygons", total=x_grid.size, unit="polygons", disable=not enable_progress_bar)
    ]
    prepare(polygon)  # Prepare the geometry for faster intersection checks
    
    # Filter polygons that intersect with the geometry
    grid_polygons = [grid_polygon for grid_polygon in tqdm(grid_polygons, desc="Filtering polygons", unit="polygons", disable=not enable_progress_bar) if polygon.intersects(grid_polygon)]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=crs)
    
    if clip_to_polygon:
        grid_gdf = grid_gdf.intersection(polygon)
    
    return grid_gdf



@cache
def load_census_shapefile(
    level: Literal["state", "county", "zcta"] = "state",
    verify: bool = True
) -> gpd.GeoDataFrame:
    """
    Download and load a US Census TIGER shapefile for states, counties, or ZIP Code Tabulation Areas (ZCTA).

    The shapefile is downloaded from the US Census Bureau TIGER/Line website, extracted, 
    and loaded as a GeoDataFrame. The function is cached to avoid repeated downloads.

    Parameters
    ----------
    level : {'state', 'county', 'zcta'}, default='state'
        Which shapefile to download. Options are:
        - 'state': US state boundaries
        - 'county': US county boundaries
        - 'zcta': ZIP Code Tabulation Areas
    verify : bool, default=True
        Whether to verify the downloaded file's SSL certificate. Do not set to False 
        in production code.

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded shapefile as a GeoDataFrame with geometries in WGS84 (EPSG:4326).

    Raises
    ------
    ValueError
        If the level argument is not one of the allowed values.
    requests.HTTPError
        If the download fails due to network or server issues.
    zipfile.BadZipFile
        If the downloaded file is not a valid ZIP archive.
    AssertionError
        If the archive does not contain exactly one .shp file.
    """
    urls = {
        "state": "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip",
        "county": "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip",
        "zcta": "https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip"
    }
    if level not in urls:
        raise ValueError(f"level must be one of {list(urls.keys())}, got '{level}'")

    url = urls[level]

    with TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "census.zip")
        
        # Download the zip file
        with requests.get(url, stream=True, verify=verify) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract the zip file
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
            
        # Find the .shp file
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        assert len(shp_files) == 1, f"Expected exactly one .shp file in the archive, found {len(shp_files)}."        
        shp_path = os.path.join(tmpdir, shp_files[0])
        
        # Load with geopandas
        gdf = gpd.read_file(shp_path)
    
    return gdf



def _flatten_dict(d, parent_key='', sep='.'): 
    """
    Recursively flatten a nested dictionary, concatenating keys with a separator.

    Nested keys are joined with the specified separator to produce a flat dictionary 
    where each key represents the path to the value in the original nested structure.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, default=''
        The base key string to prepend to each key.
    sep : str, default='.'
        Separator to use when concatenating keys.

    Returns
    -------
    dict
        A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)