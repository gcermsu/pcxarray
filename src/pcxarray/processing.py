from functools import partial
import os
from typing import Optional, List, Dict, Any, Union
import geopandas as gpd
import pandas as pd
from pyproj import Transformer, CRS, transform
import xarray as xr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from tqdm import tqdm
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry

from concurrent.futures import ThreadPoolExecutor

from .query import pc_query
from .io import read_single_item



def prepare_data(
    items_gdf: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    crs: Union[CRS, str] = 4326,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: str = 'max',
    resampling_method: Resampling = Resampling.bilinear,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    enable_progress_bar: bool = False,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
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
    enable_time_dim : bool, optional
        Whether to enable a datetime dimension in the output (default is False).
    time_col : str, optional
        Column name for datetime in items_gdf (default is 'properties.datetime').
    time_format_str : str, optional
        Format string for parsing datetime values (default is None, which uses pandas default).
    **rioxarray_kwargs: dict, optional
        Additional keyword arguments to pass to `rioxarray.open_rasterio`.
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
        
        image = read_single_item(
            item_gs=items_full_overlap.iloc[0],
            geometry=geometry,
            bands=bands,
            crs=crs,
            **rioxarray_kwargs,
        )
        
    else: # multiple items, need to merge and reproject. 
        items_gdf = items_gdf.sort_values(by='percent_overlap', ascending=False)
        
        remaining_geom = geom_84
        remaining_area = 1.0
        selected_items = []
        while remaining_area > 0:
            item_series = items_gdf.iloc[0]
            selected_items.append(item_series)
            
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
        for item_series in tqdm(selected_items, desc='Merging tiles', unit='tiles', disable=not enable_progress_bar):
            xa = read_single_item(
                item_gs=item_series,
                geometry=geometry,
                bands=bands,
                crs=crs,
                **rioxarray_kwargs,
            )
            xa = xa.rio.clip([geometry], crs=crs, all_touched=True)
            
            if image is None:
                image = xa
            else:
                if xa.rio.crs != image.rio.crs or xa.rio.resolution() != image.rio.resolution():
                    xa = xa.rio.reproject(
                        image.rio.crs,
                        resolution=image.rio.resolution(),
                        resampling=resampling_method,
                    )
                image = merge_arrays([image, xa], method=merge_method)
    
    if target_resolution is None:
        target_resolution = image.rio.resolution()[0]
    
    image = image.rio.reproject(
        resolution=(target_resolution, target_resolution),
        resampling=resampling_method,
        dst_crs=crs,
    ).rio.clip([geometry], crs=crs, all_touched=all_touched)
    
    if enable_time_dim:
        if time_col not in items_gdf.columns:
            raise ValueError(f"Column '{time_col}' not found in items_gdf.")
        datetimes = [item_series[time_col] for item_series in selected_items]
        # if datetimes are not all the same, raise an error
        if len(set(datetimes)) != 1:
            raise ValueError(f"All items must have the same '{time_col}' value to enable datetime dimension.")
        
        datetime = datetimes[0]
        if isinstance(datetime, str):
            if time_format_str is not None:
                datetime = pd.to_datetime(datetime, format=time_format_str)
            else:
                datetime = pd.to_datetime(datetime)
        
        image = image.expand_dims(time=[datetime])
    
    return image



def query_and_prepare(
    collections: Union[str, List[str]],
    geometry: BaseGeometry,
    crs: Union[CRS, str] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    return_in_wgs84: bool = False,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: str = 'max',
    resampling_method: Resampling = Resampling.bilinear,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    enable_progress_bar: bool = False,
    return_items: bool = False,
    query_kwargs: Optional[Dict[str, Any]] = None,
    rioxarray_kwargs: Optional[Dict[str, Any]] = None
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
        return_in_wgs84=return_in_wgs84,
        **query_kwargs if query_kwargs is not None else {}
    )
    
    image = prepare_data(
        items_gdf=items_gdf,
        geometry=geometry,
        crs=crs,
        bands=bands,
        target_resolution=target_resolution,
        merge_method=merge_method,
        resampling_method=resampling_method,
        enable_time_dim=enable_time_dim,
        time_col=time_col,
        time_format_str=time_format_str,
        enable_progress_bar=enable_progress_bar,
        all_touched=all_touched,
        **rioxarray_kwargs if rioxarray_kwargs is not None else {}
    )
    
    if not return_items:
        return image
    else:
        return image, items_gdf



def prepare_timeseries(
    items_gdf: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    crs: Union[CRS, str] = 4326,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: str = 'max',
    resampling_method: Resampling = Resampling.bilinear,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    max_workers: int = 1,
    enable_progress_bar: bool = True,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    
    if max_workers == 1:
        das = []
        for _, group in tqdm(
            items_gdf.groupby(time_col),
            desc="Processing items",
            unit="timestep",
            disable=not enable_progress_bar,
        ):
            da = prepare_data(
                items_gdf=group,
                geometry=geometry,
                crs=crs,
                bands=bands,
                target_resolution=target_resolution,
                all_touched=all_touched,
                merge_method=merge_method,
                resampling_method=resampling_method,
                enable_time_dim=True,
                time_col= time_col,
                time_format_str=time_format_str,
                **rioxarray_kwargs,
            )

            das.append(da)
    
    else:
        if max_workers == -1:
            max_workers = os.cpu_count()
        
        worker = partial(prepare_data,
            geometry=geometry,
            crs=crs,
            bands=bands,
            target_resolution=target_resolution, 
            all_touched=all_touched,
            merge_method=merge_method,
            resampling_method=resampling_method,
            enable_time_dim=True,
            time_col= time_col,
            time_format_str=time_format_str,
            **rioxarray_kwargs,           
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            groups = list(items_gdf.groupby(time_col))
            
            das = list(
                tqdm(
                    executor.map(lambda x: worker(x[1]), groups),
                    desc="Processing items",
                    unit="timestep",
                    total=len(groups),
                    disable=not enable_progress_bar,
                )
            )
    
    return xr.concat(das, dim='time')