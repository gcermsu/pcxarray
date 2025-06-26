import multiprocessing
import requests
from typing import Optional, List, Dict, Any, Union

import shapely
from pystac import Item
import pystac_client
import planetary_computer
import geopandas as gpd
from pyproj import Transformer, CRS, transform
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from tqdm import tqdm
from shapely.ops import transform
from shapely import from_geojson, to_geojson
from warnings import warn
from .utils import flatten_dict


class TimeoutSession(requests.Session):
    """
    A requests.Session subclass that sets a default timeout for HTTP requests.
    """
    def request(self, *args, **kwargs):
        """
        Make a request with a default timeout of 5 seconds if not specified.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the parent request method.
        **kwargs : dict
            Keyword arguments passed to the parent request method.

        Returns
        -------
        requests.Response
            The response object.
        """
        kwargs.setdefault("timeout", 5)  # Seconds: HTTP-level timeout
        return super().request(*args, **kwargs)


def _pc_query_worker(search_kwargs: Dict[str, Any], queue: multiprocessing.Queue):
    """
    Worker function to perform a Planetary Computer STAC search in a separate process.

    Parameters
    ----------
    search_kwargs : dict
        Arguments passed to Client.search().
    queue : multiprocessing.Queue
        Queue to put the result or exception.
    """
    # Inject timeout-aware session
    pystac_client.client.requests = TimeoutSession()

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
        )

        search = catalog.search(**search_kwargs)
        items = list(search.items())
        queue.put(items)
        
    except Exception as e:
        queue.put(e)


def safe_pc_search(
    search_kwargs: Dict[str, Any],
    timeout: float = 120.0
) -> Optional[List[Item]]:
    """
    Executes a Planetary Computer STAC search query with enforced timeout and safe interruption.

    Parameters
    ----------
    search_kwargs : dict
        Arguments passed to Client.search(). Example:
        {
            "collections": ["naip"],
            "intersects": {...},
            "datetime": "2000-01-01/2025-01-01"
        }
    timeout : float, optional
        Wall-clock timeout in seconds (default is 120 seconds).

    Returns
    -------
    Optional[List[Item]]
        List of STAC Items if successful, or None on timeout/failure.

    Raises
    ------
    TimeoutError
        If the query exceeds the specified timeout.
    Exception
        If an error occurs during the query.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_pc_query_worker, args=(search_kwargs, queue))

    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"Planetary Computer STAC query timed out after {timeout} seconds.")

    result = queue.get()
    if isinstance(result, Exception):
        raise result

    return result


def pc_query(
    collections: Union[str, List[str]],
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Union[CRS, str] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    query_kwargs: Optional[Dict[str, Any]] = None,
    return_in_wgs84: bool = False,
) -> gpd.GeoDataFrame:
    """
    Query the Planetary Computer STAC API and return results as a GeoDataFrame.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry.
    crs : Union[CRS, str], optional
        Coordinate reference system of the input geometry (default is 4326).
    datetime : str, optional
        Date/time range for the query (default is '2000-01-01/2025-01-01').
    query_kwargs : dict, optional
        Additional query parameters to pass to the search.
    return_in_wgs84 : bool, optional
        If True, return results in WGS84 (EPSG:4326). Otherwise, return in the input CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of the query results.

    Raises
    ------
    Warning
        If no items are found for the given query.
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
    aoi = to_geojson(geom_84).replace("'", '"')
    
    pystac_items = safe_pc_search({
        "collections": collections if isinstance(collections, list) else [collections],
        "intersects": aoi,
        "datetime": datetime,
    } | (query_kwargs or {})) # Merge with any additional query parameters
    
    items = []
    for item in pystac_items:
        item_dict = item.to_dict()
        geometry = from_geojson(item_dict.pop('geometry').__str__().replace('\'', '"'))
        item_dict['geometry'] = geometry
        items.append(flatten_dict(item_dict))
    
    if len(items) == 0:
        warn("No items found for the given query. Returning empty GeoDataFrame.")
        items_gdf = gpd.GeoDataFrame(columns=['geometry'])
    else:
        items_gdf = gpd.GeoDataFrame(items)
    
    items_gdf = items_gdf.set_crs(4326) # by default, planetary computer returns items in WGS84
    if not return_in_wgs84:
        items_gdf = items_gdf.to_crs(crs)
    
    return items_gdf


def prepare_data(
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Union[CRS, str] = 4326,
    items_gdf = None,
    masked = False,
    chunks: Optional[Dict[str, Any]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: str = 'max',
    resampling_method: Resampling = Resampling.bilinear,
    enable_progress_bar: bool = False,
):
    """
    Prepare and merge raster data from Planetary Computer query results.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry.
    crs : Union[CRS, str], optional
        Coordinate reference system for the output (default is 4326).
    items_gdf : geopandas.GeoDataFrame
        GeoDataFrame of items to process.
    masked : bool, optional
        Whether to mask the raster data (default is False).
    chunks : dict, optional
        Chunking options for dask/xarray (default is None).
    target_resolution : float, optional
        Target resolution for the output raster (default is None).
    all_touched : bool, optional
        Whether to include all pixels touched by the geometry (default is False).
    merge_method : str, optional
        Method to use when merging arrays (default is 'max').
    resampling_method : rasterio.enums.Resampling, optional
        Resampling method to use (default is Resampling.bilinear).
    enable_progress_bar : bool, optional
        Whether to display a progress bar during merging (default is False).

    Returns
    -------
    xarray.DataArray
        The prepared raster data as an xarray DataArray.
    """
    transformer = Transformer.from_crs(
        crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )
    geom_84 = transform(
        transformer.transform,
        geometry
    )
    
    items_gdf['percent_overlap'] = items_gdf.geometry.apply(lambda x: x.intersection(geom_84).area / geom_84.area)
    items_full_overlap = items_gdf[items_gdf['percent_overlap'] == 1.0]

    if len(items_full_overlap) > 1: # single item, no need to merge
        
        url = items_full_overlap.iloc[0]['assets.image.href']
        signed_url = planetary_computer.sign(url)
        image = rxr.open_rasterio(signed_url, masked=masked, chunks=chunks).rio.clip_box(*geometry.bounds, crs=crs).rio.clip([geometry], crs=crs)
        
    else: # multiple items, need to merge and reproject. 
        items_gdf = items_gdf.sort_values(by='percent_overlap', ascending=False)
        
        remaining_geom = geom_84
        remaining_area = 1.0
        urls = []
        while remaining_area > 0:
            item_series = items_gdf.iloc[0]
            url = item_series['assets.image.href']
            urls.append(url)
            
            intersection = item_series.geometry.intersection(remaining_geom)
            remaining_geom = remaining_geom.difference(intersection)
            remaining_area = remaining_geom.area / geom_84.area
            if remaining_area == 0:
                break
            
            # remove item_series from items_gdf
            items_gdf = items_gdf.iloc[1:]
            if len(items_gdf) == 0:
                break
            
            # now, recalculate the percent overlap for the remaining items
            items_gdf['percent_overlap'] = items_gdf.geometry.apply(lambda x: x.intersection(remaining_geom).area / remaining_area)
            items_gdf = items_gdf.sort_values(by='percent_overlap', ascending=False)
        
        image = None
        for url in tqdm(urls, desc='Merging tiles', unit='tiles', disable=not enable_progress_bar):
            signed_url = planetary_computer.sign(url)
            xa = rxr.open_rasterio(signed_url, masked=masked, chunks=chunks).rio.clip_box(*geometry.bounds, crs=crs)
            xa = xa.rio.clip([geometry], crs=crs, all_touched=True)
            
            if image is None:
                image = xa
            else:
                image = merge_arrays([image, xa], method=merge_method)
    
    if target_resolution is None:
        target_resolution = image.rio.resolution()[0]
    
    image = image.rio.reproject(
        resolution=(target_resolution, target_resolution),
        resampling=resampling_method,
        dst_crs=crs,
    ).rio.clip([geometry], crs=crs, all_touched=all_touched)
    
    return image


def query_and_prepare(
    collections: Union[str, List[str]],
    geometry: shapely.geometry.base.BaseGeometry,
    crs: Union[CRS, str] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    query_kwargs: Optional[Dict[str, Any]] = None,
    return_in_wgs84: bool = False,
    masked: bool = False,
    chunks: Optional[Dict[str, Any]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    return_items: bool = False
) -> Union[gpd.GeoDataFrame, tuple]:
    """
    Query the Planetary Computer and prepare raster data in a single step.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry.
    crs : Union[CRS, str], optional
        Coordinate reference system for the input/output (default is 4326).
    datetime : str, optional
        Date/time range for the query (default is '2000-01-01/2025-01-01').
    query_kwargs : dict, optional
        Additional query parameters to pass to the search.
    return_in_wgs84 : bool, optional
        If True, return results in WGS84 (EPSG:4326). Otherwise, return in the input CRS.
    masked : bool, optional
        Whether to mask the raster data (default is False).
    chunks : dict, optional
        Chunking options for dask/xarray (default is None).
    target_resolution : float, optional
        Target resolution for the output raster (default is None).
    all_touched : bool, optional
        Whether to include all pixels touched by the geometry (default is False).
    return_items : bool, optional
        If True, also return the items GeoDataFrame (default is False).

    Returns
    -------
    xarray.DataArray or tuple
        The prepared raster data, and optionally the items GeoDataFrame.
    """
    items_gdf = pc_query(
        collections=collections,
        geometry=geometry,
        crs=crs,
        datetime=datetime,
        query_kwargs=query_kwargs,
        return_in_wgs84=return_in_wgs84
    )
    
    image = prepare_data(
        geometry=geometry,
        crs=crs,
        items_gdf=items_gdf,
        masked=masked,
        chunks=chunks,
        target_resolution=target_resolution,
        all_touched=all_touched
    )
    
    if not return_items:
        return image
    else:
        return image, items_gdf
