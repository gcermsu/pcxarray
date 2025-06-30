from functools import partial
import os
from typing import Optional, List, Dict, Any, Union
from warnings import warn
import geopandas as gpd
import pandas as pd
from pyproj import Transformer, CRS, transform
import xarray as xr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from tqdm import tqdm
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from .query import pc_query
from .io import read_single_item



def prepare_data(
    items_gdf: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    crs: Union[CRS, str] = 4326,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[Union[float, int]] = None,
    all_touched: bool = False,
    merge_method: str = 'first',
    resampling_method: Resampling = Resampling.bilinear,
    reproject_first: bool = False,
    reproject_num_threads: int = 1,
    reproject_mem_limit: int = 64,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    enable_progress_bar: bool = False,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
):
    """
    Prepare and merge raster data from Planetary Computer query results.

    This function selects the minimum set of STAC items needed to cover a given geometry,
    reads and mosaics raster tiles, and handles reprojection, resampling, and merging.
    Items are selected using a greedy algorithm to minimize the number of tiles while
    ensuring complete coverage. When a single item fully covers the geometry, no merging
    is performed for efficiency.

    Parameters
    ----------
    items_gdf : geopandas.GeoDataFrame
        GeoDataFrame of STAC items to process.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry in the target CRS.
    crs : Union[CRS, str], optional
        Coordinate reference system for the output (default is 4326).
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float, optional
        Target pixel size for the output raster in units of the CRS. If None, uses 
        the native resolution of the first item (default is None).
    all_touched : bool, optional
        Whether to include all pixels touched by the geometry during final clipping 
        (default is False).
    merge_method : str, optional
        Method to use when merging overlapping arrays. Options include 'first', 'last', 
        'min', 'max', 'mean', 'sum' (default is 'first').
    resampling_method : rasterio.enums.Resampling, optional
        Resampling method to use for reprojection (default is Resampling.bilinear).
    reproject_first : bool, optional
        If True, reproject and/or resample tiles to the desired resolution/crs 
        prior to merging tiles when multiple items require merging. This can be 
        useful for reducing memory footprint when merging large rasters where
        the target pixel size is much larger (coarser) than the native resolution, 
        but it may result in a loss of spatial accuracy (default is False).
    reproject_num_threads: int, optional
        Number of threads to use for reprojection operations (default is 1, -1 uses
        all available CPUs).
    reproject_mem_limit: int, optional
        Memory limit in MB for reprojection operations. Larger values allow for
        larger chunks to be processed in parallel, but may increase memory usage
        (default is 64).
    enable_time_dim : bool, optional
        If True, add a time dimension to the output. All selected items must have 
        the same datetime value (default is False).
    time_col : str, optional
        Column name for datetime in items_gdf (default is 'properties.datetime').
    time_format_str : str, optional
        Format string for parsing datetime values (default is None, uses pandas default).
    enable_progress_bar : bool, optional
        Whether to display a progress bar during tile merging (default is False).
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        The prepared raster data as an xarray DataArray, optionally with a time dimension.

    Raises
    ------
    ValueError
        If enable_time_dim is True but time_col is not found in items_gdf, or if
        selected items have different datetime values when enable_time_dim is True.
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

    selected_items = []
    if len(items_full_overlap) >= 1: # single item, no need to merge
        
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
                if reproject_first: 
                    if target_resolution is None:
                        warn(f"reproject_first is True, but target_resolution is None. Using the native resolution of the first item: {image.rio.resolution()}.")
                        target_resolution = image.rio.resolution()[0]
                    
                    # NOTE: In an ideal world, rioxarray's underlying reprojection
                    # function would support dask arrays for parallel processing 
                    # and lazy evaluation, but currently it relies on rasterio's
                    # warp module which is not dask-aware. As such, we have to pass
                    # in the number of threads and memory limit directly. According
                    # to the rioxarray documentation (https://corteva.github.io/rioxarray/html/examples/reproject.html)
                    # odc=geo and pyresample packages offer such functionality, but
                    # the documentation is relatively scant and unclear as to how
                    # this is implemented in practice. For now, this is a stopgap,
                    # but in the future we may want to explore either utilizing
                    # these packages or implementing our own parallel reprojection
                    # functionality that utilizes dask for lazy evaluation.                    
                    
                    # reproject to target resolution
                    image = image.rio.reproject(
                        crs,
                        resolution=(target_resolution, target_resolution),
                        resampling=resampling_method,
                        num_threads=reproject_num_threads,
                        warp_mem_limit=reproject_mem_limit if reproject_mem_limit > 1 else os.cpu_count(),
                    )

            else:
                if xa.rio.crs != image.rio.crs or xa.rio.resolution() != image.rio.resolution():
                    xa = xa.rio.reproject(
                        image.rio.crs,
                        resolution=image.rio.resolution(),
                        resampling=resampling_method,
                    )
                image = merge_arrays([image, xa], method=merge_method)
    
    if image is None:
        raise ValueError("No valid image data could be processed from the selected items.")
    
    if not reproject_first and (image.rio.crs != crs or image.rio.resolution() != (target_resolution, target_resolution)):
        if target_resolution is None:
            target_resolution = image.rio.resolution()[0]
        
        image = image.rio.reproject(
            crs,
            resolution=(target_resolution, target_resolution),
            resampling=resampling_method,
            num_threads=reproject_num_threads,
            warp_mem_limit=reproject_mem_limit if reproject_mem_limit > 1 else os.cpu_count(),
        )
    
    image = image.rio.clip([geometry], crs=crs, all_touched=all_touched)
    
    if enable_time_dim:
        if time_col is None:
            raise ValueError("time_col cannot be None when enable_time_dim is True.")
        
        if len(items_full_overlap) >= 1:
            if time_col not in items_full_overlap.columns:
                raise ValueError(f"Column '{time_col}' not found in items_gdf.")
            datetime = items_full_overlap.iloc[0][time_col]
        else:
            # Check if time_col exists in the first selected item
            if not selected_items or time_col not in selected_items[0].index:
                raise ValueError(f"Column '{time_col}' not found in items.")
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
            # round to milliseconds for compatibility with netcdf/zarr
            datetime = datetime.round('ms').tz_localize(None)
            datetime = np.datetime64(datetime)
        
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
    merge_method: str = 'first',
    resampling_method: Resampling = Resampling.bilinear,
    reproject_first: bool = False,
    reproject_num_threads: int = 1,
    reproject_mem_limit: int = 64,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    enable_progress_bar: bool = False,
    return_items: bool = False,
    query_kwargs: Optional[Dict[str, Any]] = None,
    rioxarray_kwargs: Optional[Dict[str, Any]] = None
) -> Union[xr.DataArray, tuple]:
    """
    Query the Planetary Computer and prepare raster data in a single step.

    This function combines a STAC API query and raster preparation pipeline. It queries
    the Planetary Computer for items matching the given geometry, date range, and collections,
    then reads, merges, and processes the raster data. Optionally returns the items GeoDataFrame.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search within the Planetary Computer catalog.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry.
    crs : Union[CRS, str], optional
        Coordinate reference system for the input/output (default is 4326).
    datetime : str, optional
        Date/time range for the query in ISO 8601 format or interval
        (default is '2000-01-01/2025-01-01').
    return_in_wgs84 : bool, optional
        If True, return results in WGS84 (EPSG:4326). Otherwise, return in the input CRS
        (default is False).
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float, optional
        Target pixel size for the output raster in units of the CRS
        (default is None, uses native resolution).
    all_touched : bool, optional
        Whether to include all pixels touched by the geometry during clipping
        (default is False).
    merge_method : str, optional
        Method to use when merging overlapping arrays. Options include 'first', 'last',
        'min', 'max', 'mean', 'sum' (default is 'first').
    resampling_method : rasterio.enums.Resampling, optional
        Resampling method to use for reprojection (default is Resampling.bilinear).
    reproject_first : bool, optional
        If True, reproject and/or resample tiles to the desired resolution/crs prior 
        to merging tiles when multiple items require merging. This can be useful 
        for reducing memory footprint when merging large rasters where the target
        pixel size is much larger (coarser) than the native resolution, but it may
        result in a loss of spatial accuracy (default is False).
    reproject_num_threads: int, optional
        Number of threads to use for reprojection operations (default is 1, -1 uses 
        all available CPUs).
    reproject_mem_limit: int, optional
        Memory limit in MB for reprojection operations. Larger values allow for 
        larger chunks to be processed in parallel, but may increase memory usage
        (default is 64).
    enable_time_dim : bool, optional
        If True, add a time dimension to the output (default is False).
    time_col : str, optional
        Column name for datetime in items_gdf (default is 'properties.datetime').
    time_format_str : str, optional
        Format string for parsing datetime values (default is None, uses pandas default).
    enable_progress_bar : bool, optional
        Whether to display a progress bar during merging (default is False).
    return_items : bool, optional
        If True, also return the items GeoDataFrame (default is False).
    query_kwargs : dict, optional
        Additional query parameters to pass to the STAC search.
    rioxarray_kwargs : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray or tuple
        The prepared raster data. If return_items is True, returns a tuple of
        (DataArray, GeoDataFrame).

    Notes
    -----
    This is a convenience function that combines pc_query() and prepare_data().
    For more control over the process, use those functions separately.
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
        reproject_first=reproject_first,
        reproject_num_threads=reproject_num_threads,
        reproject_mem_limit=reproject_mem_limit,
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
    merge_method: str = 'first',
    resampling_method: Resampling = Resampling.bilinear,
    reproject_first: bool = False,
    reproject_num_threads: int = 1,
    reproject_mem_limit: int = 64,
    time_col: str = 'properties.datetime',
    time_format_str: Optional[str] = None,
    ignore_time_component: bool = True,
    chunks: Optional[Dict[str, int]] = None,
    max_workers: int = 1,
    enable_progress_bar: bool = True,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    """
    Prepare a time series of raster data from a GeoDataFrame of STAC items.

    This function groups items by time, reads and merges rasters for each timestep,
    and concatenates them into a single DataArray along the time dimension. Supports
    parallel processing and chunking for large datasets.

    Parameters
    ----------
    items_gdf : geopandas.GeoDataFrame
        GeoDataFrame of STAC items to process.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry in the target CRS.
    crs : Union[CRS, str], optional
        Coordinate reference system for the output (default is 4326).
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float, optional
        Target pixel size for the output raster (default is None, uses native resolution).
    all_touched : bool, optional
        Whether to include all pixels touched by the geometry (default is False).
    merge_method : str, optional
        Method to use when merging arrays (e.g., 'first', 'last', 'min', 'max', 'mean', 'sum')
        (default is 'first').
    resampling_method : rasterio.enums.Resampling, optional
        Resampling method to use for reprojection (default is Resampling.bilinear).
    reproject_first : bool, optional
        If True, reproject and/or resample tiles to the desired resolution/crs prior 
        to merging tiles when multiple items require merging. This can be useful for 
        reducing memory footprint when merging large rasters where the target pixel 
        size is much larger (coarser) than the native resolution, but it may result 
        in a loss of spatial accuracy (default is False).
    reproject_num_threads: int, optional
        Number of threads to use for reprojection operations (default is 1, -1 uses
        all available CPUs).
    reproject_mem_limit: int, optional
        Memory limit in MB for reprojection operations. Larger values allow for
        larger chunks to be processed in parallel, but may increase memory usage
        (default is 64).
    time_col : str, optional
        Column name for datetime in items_gdf (default is 'properties.datetime').
    ignore_time_component : bool, optional
        If True, ignore the time component and only use the date (default is True).
    time_format_str : str, optional
        Format string for parsing datetime values (default is None, uses pandas default).
    chunks : dict, optional
        Chunking options for dask/xarray (default is None).
    max_workers : int, optional
        Number of parallel workers to use (default is 1; -1 uses all available CPUs).
    enable_progress_bar : bool, optional
        Whether to display a progress bar during processing (default is True).
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        The prepared time series raster data as an xarray DataArray with a time dimension.
    """
    
    # chunk handling
    chunks_no_time = None
    chunks_no_band_time = None
    if chunks is not None and isinstance(chunks, Dict):
        # need to ensure we don't chunk by band or time dimensions when they don't exist yet
        chunks_no_time = {k: v for k, v in chunks.items() if k != 'time'}
        chunks_no_band_time = {k: v for k, v in chunks_no_time.items() if k != 'band'}
    
    if time_col not in items_gdf.columns:
        raise ValueError(f"Column '{time_col}' not found in items_gdf. Please provide a valid time column.")
    items_gdf = items_gdf.sort_values(by=time_col)
    
    if ignore_time_component:
        items_gdf[time_col] = items_gdf[time_col].dt.normalize()
    
    if max_workers == 1:
        das = []
        for _, group in tqdm(
            items_gdf.groupby(time_col),
            desc="Processing items",
            unit="timestep",
            disable=not enable_progress_bar,
        ):
            da = prepare_data(
                items_gdf=gpd.GeoDataFrame(group),
                geometry=geometry,
                crs=crs,
                bands=bands,
                target_resolution=target_resolution,
                all_touched=all_touched,
                merge_method=merge_method,
                resampling_method=resampling_method,
                reproject_first=reproject_first,
                reproject_num_threads=reproject_num_threads,
                reproject_mem_limit=reproject_mem_limit,
                enable_time_dim=True,
                time_col= time_col,
                time_format_str=time_format_str,
                enable_progress_bar=False,
                chunks=chunks_no_band_time,
                **rioxarray_kwargs,
            )
            if chunks is not None:
                da = da.chunk(chunks_no_time)

            das.append(da)
    
    else:
        if max_workers == -1:
            max_workers = os.cpu_count() or 1
        
        worker = partial(prepare_data,
            geometry=geometry,
            crs=crs,
            bands=bands,
            target_resolution=target_resolution, 
            all_touched=all_touched,
            merge_method=merge_method,
            resampling_method=resampling_method,
            reproject_first=reproject_first,
            reproject_num_threads=reproject_num_threads,
            reproject_mem_limit=reproject_mem_limit,
            enable_time_dim=True,
            time_col=time_col,
            time_format_str=time_format_str,
            enable_progress_bar=False,
            chunks=chunks_no_band_time,
            **rioxarray_kwargs,           
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            groups = list(items_gdf.groupby(time_col))
            
            futures = [executor.submit(worker, items_gdf=gpd.GeoDataFrame(group)) for _, group in groups]
            
            das = []
            with tqdm(
                as_completed(futures),
                desc="Processing items",
                unit="timestep",
                total=len(groups),
                disable=not enable_progress_bar,
            ) as progress:
                for future in progress:
                    try:
                        da = future.result()
                        if chunks is not None:
                            da = da.chunk(chunks_no_time)
                        das.append(da)
                    except Exception as e:
                        print(f"Error processing group: {e}")
                        continue
    
    da = xr.concat(das, dim='time')
    if chunks is not None:
        da = da.chunk(chunks)
        return da.sortby('time').chunk(chunks)
    
    return da.sortby('time')