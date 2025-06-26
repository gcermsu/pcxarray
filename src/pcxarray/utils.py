import os
from tempfile import TemporaryDirectory
from typing import Literal, Union
from zipfile import ZipFile
import geopandas as gpd
import numpy as np
from pyproj import CRS
import requests
from shapely.geometry import Polygon
from shapely import prepare
from tqdm import tqdm

def create_grid(
    polygon: Polygon, 
    crs: Union[CRS, str], 
    cell_size: int = 1000,  
    enable_progress_bars: bool = False,
    clip_to_polygon: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create a grid of square polygons over a given polygon geometry.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon to cover with a grid.
    crs : Union[pyproj.CRS, str]
        The coordinate reference system for the output GeoDataFrame.
    cell_size : int, optional
        The size of each grid cell along each side in the units of the CRS (default is 1000).
    enable_progress_bars : bool, optional
        Whether to display progress bars during grid creation and filtering (default is False).
    clip_to_polygon : bool, optional
        If True, grid cells will be clipped to the input polygon boundary (default is True).

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
        for x, y in tqdm(zip(x_grid.ravel(), y_grid.ravel()), desc="Creating grid polygons", total=x_grid.size, unit="polygons", disable=not enable_progress_bars)
    ]
    prepare(polygon)  # Prepare the geometry for faster intersection checks
    
    # Filter polygons that intersect with the geometry
    grid_polygons = [grid_polygon for grid_polygon in tqdm(grid_polygons, desc="Filtering polygons", unit="polygons", disable=not enable_progress_bars) if polygon.intersects(grid_polygon)]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=crs)
    
    if clip_to_polygon:
        grid_gdf = grid_gdf.intersection(polygon)
    
    return grid_gdf




def load_census_shapefile(level: Literal["state", "county"]="state", verify: bool = True) -> gpd.GeoDataFrame:
    """
    Download and load a US Census TIGER shapefile for states or counties.

    Parameters
    ----------
    level : {'state', 'county'}, optional
        Which shapefile to download: 'state' or 'county'. Default is 'state'.
    verify : bool, optional
        Whether to verify the downloaded file's integrity. Do not set to False
        in production code. Default is True.

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded shapefile as a GeoDataFrame.

    Raises
    ------
    ValueError
        If the level argument is not 'state' or 'county'.
    FileNotFoundError
        If the .shp file is not found in the extracted archive.
    AssertionError
        If more than one .shp file is found in the archive.
    """
    urls = {
        "state": "http://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip",
        "county": "http://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip",
    }
    if level not in urls:
        raise ValueError("level must be 'state' or 'county'")

    url = urls[level]

    with TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "census.zip")
        
        # Download the zip file
        with requests.get(url, stream=True, verify=False) as r:
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



def flatten_dict(d, parent_key='', sep='.'): 
    """
    Recursively flattens a nested dictionary, concatenating keys with a separator.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, optional
        The base key string to prepend to each key (default is '').
    sep : str, optional
        Separator to use when concatenating keys (default is '.').

    Returns
    -------
    dict
        A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)