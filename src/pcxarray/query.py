from time import sleep
import shapely
from pystac import Item
from pystac_client import Client
import pandas as pd
import geopandas as gpd
from pyproj import Transformer, CRS, transform
from shapely.ops import transform
from shapely import from_geojson
from shapely.geometry import box
from warnings import warn
from concurrent.futures import ThreadPoolExecutor
from .utils import _flatten_dict
from .cache import cache
from joblib import expires_after
from typing import Optional, List, Dict, Any, Union


# @cache(cache_validation_callback=expires_after(minutes=15))
def safe_pc_search(
    search_kwargs: Dict[str, Any],
    timeout: float = 300.0
) -> List[Item]:
    """
    Perform a STAC search with a wall-clock timeout using a thread.

    Executes a STAC API search in a separate thread and enforces a maximum wall-clock 
    timeout. If the search does not complete within the specified timeout, a TimeoutError 
    is raised. This is useful for preventing long-running or hanging queries from 
    blocking the main process.

    Parameters
    ----------
    search_kwargs : dict
        Dictionary of keyword arguments to pass to pystac_client.Client.search.
    timeout : float, default=300.0
        Maximum time in seconds to wait for the search to complete.

    Returns
    -------
    list of pystac.Item
        List of STAC items returned by the search.

    Raises
    ------
    concurrent.futures.TimeoutError
        If the search does not complete within the specified timeout.
    Exception
        Any exception raised during the search will be propagated.
    """
    
    def worker():
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
        )
        search = catalog.search(**search_kwargs)
        return list(search.items())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        return future.result(timeout=timeout)



@cache(cache_validation_callback=expires_after(minutes=15))
def pc_query(
    collections: Union[str, List[str]],
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Union[CRS, str, int] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    max_retries: int = 5,
    **query_kwargs: Optional[Dict[str, Any]],
) -> gpd.GeoDataFrame:
    """
    Query the Planetary Computer STAC API and return results as a GeoDataFrame.

    Searches the Planetary Computer STAC catalog for items matching the specified 
    criteria. The input geometry is transformed to WGS84 for the query, and results 
    are returned in either WGS84 or the original input CRS.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search within the Planetary Computer catalog.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry for spatial filtering.
    crs : pyproj.CRS, str or int, default=4326
        Coordinate reference system of the input geometry.
    datetime : str, default='2000-01-01/2025-01-01'
        Date/time range for temporal filtering in ISO 8601 format or interval.
    max_retries : int, default=5
        Maximum number of retries for the STAC search in case of failure.
    **query_kwargs : dict, optional
        Additional query parameters to pass to the STAC search (e.g., 'query' for 
        property filtering, 'limit' for result count limits).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the query results with flattened STAC item properties. 
        Contains a 'geometry' column with item footprints and additional columns 
        for item metadata. The 'properties.datetime' column is converted to pandas 
        datetime if present.

    Warns
    -----
    UserWarning
        If no items are found for the given query criteria.

    Notes
    -----
    - The function automatically handles CRS transformation between the input CRS and WGS84.
    - STAC item properties are flattened using dot notation (e.g., 'properties.datetime').
    - Datetime values are rounded to milliseconds for netCDF/Zarr compatibility.
    """
    transformer = Transformer.from_crs(
        crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )
    geom_84 = transform(
        transformer.transform,
        geometry,
    )
    
    aoi = geom_84.__geo_interface__
    query = {
        "collections": collections if isinstance(collections, list) else [collections],
        "intersects": aoi,
        "datetime": str(datetime), # cast datetime to string
    } | (query_kwargs or {})  # Merge with any additional query parameters
    
    retries = 0
    while True:
        try:
            # Perform the STAC search with the given query parameters
            pystac_items = safe_pc_search(query)
            break  # Exit loop if search is successful
        except Exception as e:
            if retries >= max_retries:
                raise RuntimeError(f"STAC search failed after {max_retries} retries: {type(e).__str__}: {e}") from e
            
            warn(f"STAC search failed: {type(e).__str__()}: {e}. Retrying ({retries + 1}/{max_retries})...")
            sleep(2 ** (retries + 1))  # Exponential backoff
            retries += 1
    
    items = []
    for item in pystac_items:
        item_dict = item.to_dict()
        geometry = from_geojson(item_dict.pop('geometry').__str__().replace('\'', '"'))
        if isinstance(geometry, shapely.geometry.MultiPolygon) and len(geometry.geoms) == 1:
            geometry = geometry.geoms[0]
        item_dict['geometry'] = geometry
        items.append(_flatten_dict(item_dict))
    
    if len(items) == 0:
        warn("No items found for the given query. Returning empty GeoDataFrame.")
        items_gdf = gpd.GeoDataFrame(columns=['geometry'])
    else:
        items_gdf = gpd.GeoDataFrame(items)
    
    items_gdf = items_gdf.set_crs(4326) # by default, planetary computer returns items in WGS84
    items_gdf = items_gdf.to_crs(crs) 
    
    # set datetime column if it exists
    if 'properties.datetime' in items_gdf.columns:
        items_gdf['properties.datetime'] = pd.to_datetime(items_gdf['properties.datetime'])
        # round to milliseconds for compatibility with netcdf/zarr
        items_gdf['properties.datetime'] = [t.tz_localize(None) for t in items_gdf['properties.datetime'].dt.round('ms')]
        items_gdf['properties.datetime'] = items_gdf['properties.datetime'].astype('datetime64[ms]')
        items_gdf = items_gdf.sort_values(by='properties.datetime').reset_index()
    
    return items_gdf


def get_pc_collections(reduced: bool = True, crs: Union[CRS, str, int] = 4326) -> gpd.GeoDataFrame:
    """
    Get Planetary Computer STAC collections as a GeoDataFrame.

    Parameters
    ----------
    reduced : bool, default=True
        If True, keep only essential fields in the output.
    crs : pyproj.CRS, str or int, default=4326
        Coordinate reference system for the output GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of STAC collections with geometry column.
    """
    
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    collections = list(client.get_collections())
    
    collections_data = []
    for collection in collections:
        collection_dict = _flatten_dict(collection.to_dict())
        geom = collection_dict.pop('extent.spatial.bbox', None)
        if geom is not None:
            if len(geom) == 1:
                collection_dict['geometry'] = box(*geom[0])
            else:
                geoms = []
                for g in geom:
                    geoms.append(box(*g))
                collection_dict['geometry'] = shapely.geometry.MultiPolygon(geoms)

        else:
            collection_dict['geometry'] = None
            
        if reduced:
            # Keep only essential fields
            collection_dict = {
                'id': collection_dict.get('id', ''),
                'title': collection_dict.get('title', ''),
                'description': collection_dict.get('description', ''),
                'license': collection_dict.get('license', ''),
                'geometry': collection_dict.get('geometry', None),
            }
        
        collections_data.append(collection_dict)
    
    gdf = gpd.GeoDataFrame(collections_data, crs=4326)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf

