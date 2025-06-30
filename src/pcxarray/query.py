from time import sleep
import shapely
from pystac import Item
import pystac_client
import geopandas as gpd
from pyproj import Transformer, CRS, transform
from shapely.ops import transform
from shapely import from_geojson
from warnings import warn
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .utils import _flatten_dict
from .cache import cache
from joblib import expires_after
from typing import Optional, List, Dict, Any, Union


# @cache(cache_validation_callback=expires_after(minutes=15))
def safe_pc_search(
    search_kwargs: Dict[str, Any],
    timeout: float = 120.0
) -> List[Item]:
    """
    Perform a STAC search with a wall-clock timeout using a thread.

    This function executes a STAC API search in a separate thread and enforces
    a maximum wall-clock timeout. If the search does not complete within the
    specified timeout, a TimeoutError is raised. This is useful for preventing
    long-running or hanging queries from blocking the main process.

    Parameters
    ----------
    search_kwargs : dict
        Dictionary of keyword arguments to pass to pystac_client.Client.search.
    timeout : float, optional
        Maximum time in seconds to wait for the search to complete (default is 120.0).

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
        catalog = pystac_client.Client.open(
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
    crs: Union[CRS, str] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    return_in_wgs84: bool = False,
    max_retries: int = 5,
    **query_kwargs: Optional[Dict[str, Any]],
) -> gpd.GeoDataFrame:
    """
    Query the Planetary Computer STAC API and return results as a GeoDataFrame.

    This function searches the Planetary Computer STAC catalog for items matching
    the specified criteria. The input geometry is transformed to WGS84 for the query,
    and results are returned in either WGS84 or the original input CRS.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search within the Planetary Computer catalog.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry for spatial filtering.
    crs : Union[CRS, str], optional
        Coordinate reference system of the input geometry (default is 4326).
    datetime : str, optional
        Date/time range for temporal filtering in ISO 8601 format or interval
        (default is '2000-01-01/2025-01-01').
    return_in_wgs84 : bool, optional
        If True, return results in WGS84 (EPSG:4326). If False, return in the 
        input CRS (default is False).
    max_retries : int, optional
        Maximum number of retries for the STAC search in case of failure (default is 5).
    **query_kwargs : dict, optional
        Additional query parameters to pass to the STAC search (e.g., 'query' for
        property filtering, 'limit' for result count limits).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the query results with flattened STAC item properties.
        Contains a 'geometry' column with item footprints and additional columns for
        item metadata. The 'properties.datetime' column is converted to pandas datetime
        if present.

    Warns
    -----
    UserWarning
        If no items are found for the given query criteria.

    Notes
    -----
    - The function automatically handles CRS transformation between the input CRS and WGS84
    - STAC item properties are flattened using dot notation (e.g., 'properties.datetime')
    - Datetime values are rounded to milliseconds for netCDF/Zarr compatibility
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
            
            warn(f"STAC search failed: {type(e).__str__}: {e}. Retrying ({retries + 1}/{max_retries})...")
            sleep(2 ** (retries + 1))  # Exponential backoff
            retries += 1
    
    items = []
    for item in pystac_items:
        item_dict = item.to_dict()
        geometry = from_geojson(item_dict.pop('geometry').__str__().replace('\'', '"'))
        item_dict['geometry'] = geometry
        items.append(_flatten_dict(item_dict))
    
    if len(items) == 0:
        warn("No items found for the given query. Returning empty GeoDataFrame.")
        items_gdf = gpd.GeoDataFrame(columns=['geometry'])
    else:
        items_gdf = gpd.GeoDataFrame(items)
    
    items_gdf = items_gdf.set_crs(4326) # by default, planetary computer returns items in WGS84
    if not return_in_wgs84:
        items_gdf = items_gdf.to_crs(crs)
    
    # set datetime column if it exists
    if 'properties.datetime' in items_gdf.columns:
        items_gdf['properties.datetime'] = pd.to_datetime(items_gdf['properties.datetime'])
        # round to milliseconds for compatibility with netcdf/zarr
        items_gdf['properties.datetime'] = [t.tz_localize(None) for t in items_gdf['properties.datetime'].dt.round('ms')]
        items_gdf['properties.datetime'] = items_gdf['properties.datetime'].astype('datetime64[ms]')
    
    return items_gdf


