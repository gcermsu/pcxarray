from functools import partial
import os
from typing import Literal, Optional, List, Dict, Any, Union
from warnings import warn
import geopandas as gpd
import pandas as pd
from pyproj import CRS
import xarray as xr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from tqdm import tqdm
from shapely.ops import transform, unary_union
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
import odc.geo.xr
from joblib import Parallel, delayed

from .query import pc_query
from .io import read_single_item


def lazy_merge_arrays(
    arrays: List[xr.DataArray],
    method: Literal['last', 'first', 'min', 'max', 'mean', 'sum', 'median'] = 'last',
    geometry: Optional[BaseGeometry] = None,
    crs: Optional[Union[CRS, str, int]] = None,
    resolution: Optional[Union[float, int]] = None,
    resampling_method: Union[Resampling, str] = 'nearest',
    nodata: Optional[float] = None,
) -> xr.DataArray:
    """
    Merge multiple xarray DataArrays lazily.

    Reprojects all input arrays to a common geobox and then merges them using the 
    specified method. If geometry, CRS, or resolution are not provided, they are 
    automatically determined from the input arrays using the union of geometries, 
    first CRS found, and minimum resolution respectively. Unlike rioxarray.rio.reproject, 
    this function does not trigger a computation of the dask graph.

    Parameters
    ----------
    arrays : list of xarray.DataArray
        List of georeferenced DataArrays to merge from rioxarray.
    method : {'last', 'first', 'min', 'max', 'mean', 'sum', 'median'}, default='last'
        Method for merging overlapping pixels.
    geometry : shapely.geometry.base.BaseGeometry, optional
        Target geometry for the merged array. If None, computed as the unary union 
        of all input array bounds.
    crs : pyproj.CRS, str, or int, optional
        Target coordinate reference system. If None, uses the CRS from the first 
        input array.
    resolution : float or int, optional
        Target pixel resolution in CRS units. If None, uses the minimum resolution 
        from all input arrays.
    resampling_method : rasterio.enums.Resampling or str, default 'nearest'
        Resampling method for reprojection. Can be Resampling enum or string name 
        (e.g., 'nearest', 'bilinear', 'cubic', etc.).
    nodata : float, optional
        NoData value to use for masking. If None, uses the NoData value from the 
        first input array.

    Returns
    -------
    xarray.DataArray
        Merged DataArray reprojected to the common geobox with spatial coordinates 
        and CRS information preserved.

    Raises
    ------
    ValueError
        If an unknown merge method is specified or if geometry and CRS do not produce
        a valid geobox.
    UserWarning
        If reprojection fails for any input array (the array is skipped and 
        processing continues).
        
    Notes
    -----
    - TODO: At some point, consider documenting edge cases around nodata handling.
        This seems to be the most fragile part of this package, though it's difficult
        to really nail down all the edge cases.
    """
    
    input_dtypes = [da.dtype for da in arrays]
    if len(set(input_dtypes)) > 1:
        raise ValueError(
            "All input arrays must have the same dtype for merging. "
            f"Found differing dtypes: {set(input_dtypes)}"
        )
    input_dtype = input_dtypes[0]
    
    if geometry is None:
        geometry = unary_union([box(*da.rio.bounds()) for da in arrays])
    if crs is None:
        crs = arrays[0].rio.crs
    if resolution is None:
        resolution = min([min(abs(r) for r in da.rio.resolution()) for da in arrays])
    
    if isinstance(resampling_method, Resampling):
        resampling_method = resampling_method.name.lower()
    
    # reproject all arrays to a common geobox
    nodata = nodata if nodata is not None else arrays[0].rio.nodata
    if nodata is not None and not np.isnan(nodata):
        arrays = [da.where(da != nodata) for da in arrays]  # mask nodata values
    
    geobox = GeoBox.from_geopolygon(
        Geometry(geometry, crs=crs),
        resolution=resolution,
    )
        
    reprojected_arrays = []
    for da in arrays:
        try:
            reprojected_arrays.append(da.odc.reproject(
                how=geobox,
                resampling=resampling_method,
            ))
        except Exception as e:
            warn(f"Reprojection failed for {da.name} with error: {e}. Skipping this array.")
            continue
    
    arrays = xr.align(*reprojected_arrays, join='exact')
    stacked = xr.concat(arrays, dim='merge_dim')
    # if nodata is not None and not np.isnan(nodata):
    #     stacked = stacked.where(stacked != nodata)
    
    if method == 'last':
        filled = stacked.ffill(dim='merge_dim')
        result = filled.isel(merge_dim=-1)
    elif method == 'first':
        filled = stacked.bfill(dim='merge_dim')
        result = filled.isel(merge_dim=0)
    elif method == 'min':
        result = stacked.min(dim='merge_dim', skipna=True)
    elif method == 'max':
        result = stacked.max(dim='merge_dim', skipna=True)
    elif method == 'mean':
        result = stacked.mean(dim='merge_dim', skipna=True)
    elif method == 'sum':
        result= stacked.sum(dim='merge_dim', skipna=True)
    elif method == 'median':
        result = stacked.median(dim='merge_dim', skipna=True)
    else:
        raise ValueError(f"Unknown merge method: {method}")
    
    result = result.rio.write_nodata(nodata)
    
    if nodata is not None and not np.isnan(nodata):
        result = result.fillna(nodata)
    
    if result.dtype != input_dtype:
        result = result.astype(input_dtype)

    return result



def prepare_data(
    items_gdf: gpd.GeoDataFrame,
    geometry: Optional[BaseGeometry] = None,
    crs: Optional[Union[CRS, str, int]] = None,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[Union[float, int]] = None,
    all_touched: bool = False,
    merge_method: Literal['last', 'first', 'min', 'max', 'mean', 'sum', 'median'] = 'last',
    resampling_method: Union[Resampling, str] = 'nearest',
    chunks: Union[str, Dict[str, int], None] = None,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    max_workers: int = 1,
    enable_progress_bar: bool = False,
    **rioxarray_kwargs: Optional[Dict[str, Any]],
) -> xr.DataArray:
    """
    Prepare and merge raster data from Planetary Computer query results.

    Selects the minimum set of STAC items needed to cover a given geometry, reads 
    and mosaics raster tiles, and handles reprojection, resampling, and merging. 
    Items are selected using a greedy algorithm to minimize the number of tiles 
    while ensuring complete coverage. When a single item fully covers the geometry, 
    no merging is performed for efficiency.

    Parameters
    ----------
    items_gdf : geopandas.GeoDataFrame
        GeoDataFrame of STAC items to process.
    geometry : shapely.geometry.base.BaseGeometry, optional
        Area of interest geometry in the target CRS. If None, uses the union of 
        all geometries in items_gdf.
    crs : pyproj.CRS, str or int, optional
        Coordinate reference system for the output. If None, uses the CRS from items_gdf.
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float or int, optional
        Target pixel size for the output raster in units of the CRS. If None, uses 
        the native resolution of the first item.
    all_touched : bool, default=False
        Whether to include all pixels touched by the geometry during final clipping.
    merge_method : {'last', 'first', 'min', 'max', 'mean', 'sum', 'median'}, default='last'
        Method to use when merging overlapping arrays.
    resampling_method : rasterio.enums.Resampling or str, default='nearest'
        Resampling method to use for reprojection.
    chunks : str or dict, optional
        Chunking options for dask/xarray.
    enable_time_dim : bool, default=False
        If True, add a time dimension to the output. All selected items must have 
        the same datetime value.
    time_col : str, default='properties.datetime'
        Column name for datetime in items_gdf.
    time_format_str : str, optional
        Format string for parsing datetime values.
    max_workers : int, default=1
        Number of parallel workers to use (-1 uses all available CPUs).
    enable_progress_bar : bool, default=False
        Whether to display a progress bar during tile merging.
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        The prepared raster data as an xarray DataArray, optionally with a time 
        dimension.

    Raises
    ------
    ValueError
        If enable_time_dim is True but time_col is not found in items_gdf, or if 
        selected items have different datetime values when enable_time_dim is True.
    """
    
    if isinstance(resampling_method, Resampling):
        resampling_method = resampling_method.name.lower()
    
    if crs is not None and items_gdf.crs != crs:
        items_gdf = items_gdf.to_crs(crs)
    elif crs is None and items_gdf.crs is not None:
        crs = items_gdf.crs
    elif crs is None:
        raise ValueError("CRS must be provided if items_gdf has no CRS.")

    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    if geometry is None:
        geometry = items_gdf.union_all()
    
    items_gdf['percent_overlap'] = items_gdf.geometry.apply(lambda x: x.intersection(geometry).area / geometry.area)
    items_full_overlap = items_gdf[items_gdf['percent_overlap'] == 1.0]

    selected_items = []
    if len(items_full_overlap) >= 1: # single item, no need to merge
        da = None
        for _, item in items_full_overlap.iterrows():
            try:
                da = read_single_item(
                    item_gs=item,
                    geometry=geometry.buffer(1) if crs.is_projected else geometry, # buffer slightly to avoid edge cases
                    bands=bands,
                    chunks=chunks,
                    all_touched=True,
                    clip_to_geometry=False, # Wait to clip until after merging
                    crs=crs,
                    **rioxarray_kwargs,
                )
                break # Exit loop if successful
            except Exception as e:
                warn(f"Failed to read item {item['id']} with error: {e}. Skipping this item.")
                continue
        
        if da is None:
            raise ValueError("Could not read any items that supposedly had full overlap with the geometry. Try checking the geometry and its CRS.")
        
        da = da.odc.reproject(
            how=GeoBox.from_geopolygon(
                Geometry(geometry, crs=crs),
                resolution=target_resolution if target_resolution is not None else abs(min(da.rio.resolution())),
            ),
            resampling=resampling_method,
        )
        
    else: # multiple items, need to merge and reproject. 
        items_gdf = items_gdf.sort_values(by='percent_overlap', ascending=False)
        
        remaining_geom = geometry
        remaining_area = 1.0
        selected_items = []
        while remaining_area > 0:
            item_series = items_gdf.iloc[0]
            selected_items.append(item_series)
            
            intersection = item_series.geometry.intersection(remaining_geom)
            remaining_geom = remaining_geom.difference(intersection)
            remaining_area = remaining_geom.area / geometry.area
            if remaining_area == 0:
                break
            
            # remove item_series from items_gdf
            items_gdf = items_gdf.iloc[1:]
            if len(items_gdf) == 0:
                break
            
            # now, recalculate the percent overlap for the remaining items
            items_gdf['percent_overlap'] = items_gdf.geometry.apply(lambda x: x.intersection(remaining_geom).area / remaining_area)
            items_gdf = items_gdf.sort_values(by='percent_overlap', ascending=False)
        
        das = []
        
        def safe_read_item(item_series):
            try:
                return read_single_item(
                    item_gs=item_series,
                    geometry=geometry.buffer(1) if crs.is_projected else geometry,
                    bands=bands,
                    chunks=chunks,
                    all_touched=True,
                    clip_to_geometry=False, # Wait to clip until after merging
                    crs=crs,
                    **rioxarray_kwargs,
                )
            except Exception as e:
                warn(f"Failed to read item {item_series['id']} with error: {e}. Skipping this item.")
                return None
        
        if max_workers == -1:
            max_workers = os.cpu_count() or 1
        max_workers = min(max_workers, len(selected_items))  # limit to number of items
        
        das = Parallel(n_jobs=max_workers)(
            delayed(safe_read_item)(item_series)
            for item_series in tqdm(
                selected_items, 
                desc='Merging tiles' if chunks is None else 'Constructing dask computation graph',
                unit='tiles', 
                disable=not enable_progress_bar
            )
        )
        
        das = [da for da in das if da is not None]
        if len(das) == 0:
            raise ValueError("Could not read any items that supposedly had partial overlap with the geometry. Try checking the geometry and its CRS.")

        da = lazy_merge_arrays(
            das,
            method=merge_method,
            geometry=geometry,
            crs=crs,
            resolution=target_resolution,
            resampling_method=resampling_method,
        )
    
    da = da.odc.crop(
        Geometry(geometry, crs=crs),
        apply_mask=True,
        all_touched=all_touched,
    )
            
    # return da
    if chunks is not None and da.chunks != chunks:
        da = da.chunk(chunks)
    
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
        
        da = da.expand_dims(time=[datetime])
    
    return da



def query_and_prepare(
    collections: Union[str, List[str]],
    geometry: BaseGeometry,
    crs: Union[CRS, str, int] = 4326,
    datetime: str = "2000-01-01/2025-01-01",
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: Literal['last', 'first', 'min', 'max', 'mean', 'sum', 'median'] = 'last',
    resampling_method: Union[Resampling, str] = 'nearest',
    chunks: Union[str, Dict[str, int], None] = None,
    enable_time_dim: bool = False,
    time_col: Optional[str] = 'properties.datetime',
    time_format_str: Optional[str] = None,
    max_workers: int = 1,
    enable_progress_bar: bool = False,
    return_items: bool = False,
    query_kwargs: Optional[Dict[str, Any]] = None,
    rioxarray_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[xr.DataArray, tuple]:
    """
    Query the Planetary Computer and prepare raster data in a single step.

    Combines a STAC API query and raster preparation pipeline. It queries the Planetary 
    Computer for items matching the given geometry, date range, and collections, 
    then reads, merges, and processes the raster data. Optionally returns the items 
    GeoDataFrame.

    Parameters
    ----------
    collections : str or list of str
        Collection(s) to search within the Planetary Computer catalog.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry.
    crs : pyproj.CRS, str or int, default=4326
        Coordinate reference system for the input/output.
    datetime : str, default='2000-01-01/2025-01-01'
        Date/time range for the query in ISO 8601 format or interval.
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float or int, optional
        Target pixel size for the output raster in units of the CRS.
    all_touched : bool, default=False
        Whether to include all pixels touched by the geometry during clipping.
    merge_method : {'last', 'first', 'min', 'max', 'mean', 'sum', 'median'}, default='last'
        Method to use when merging overlapping arrays.
    resampling_method : rasterio.enums.Resampling or str, default='nearest'
        Resampling method to use for reprojection.
    chunks : str or dict, optional
        Chunking options for dask/xarray.
    enable_time_dim : bool, default=False
        If True, add a time dimension to the output.
    time_col : str, default='properties.datetime'
        Column name for datetime in items_gdf.
    time_format_str : str, optional
        Format string for parsing datetime values.
    max_workers : int, default=1
        Number of parallel workers to use (-1 uses all available CPUs).
    enable_progress_bar : bool, default=False
        Whether to display a progress bar during merging.
    return_items : bool, default=False
        If True, also return the items GeoDataFrame.
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
        **query_kwargs if query_kwargs is not None else {}
    )
    
    image = prepare_data(
        items_gdf=items_gdf,
        geometry=geometry,
        crs=crs,
        bands=bands,
        target_resolution=target_resolution,
        all_touched=all_touched,
        merge_method=merge_method,
        resampling_method=resampling_method,
        chunks=chunks,
        enable_time_dim=enable_time_dim,
        time_col=time_col,
        time_format_str=time_format_str,
        max_workers=max_workers,
        enable_progress_bar=enable_progress_bar,
        **rioxarray_kwargs if rioxarray_kwargs is not None else {}
    )
    
    if not return_items:
        return image
    else:
        return image, items_gdf



def prepare_timeseries(
    items_gdf: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    crs: Union[CRS, str, int] = 4326,
    bands: Optional[List[Union[str, int]]] = None,
    target_resolution: Optional[float] = None,
    all_touched: bool = False,
    merge_method: Literal['last', 'first', 'min', 'max', 'mean', 'sum', 'median'] = 'last',
    resampling_method: Union[Resampling, str] = 'nearest',
    chunks: Optional[Dict[str, int]] = None,
    time_col: str = 'properties.datetime',
    time_format_str: Optional[str] = None,
    ignore_time_component: bool = True,
    max_workers: int = 1,
    enable_progress_bar: bool = True,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    """
    Prepare a time series of raster data from a GeoDataFrame of STAC items.

    Groups items by time, reads and merges rasters for each timestep, and concatenates 
    them into a single DataArray along the time dimension. Supports parallel processing 
    and chunking for large datasets.

    Parameters
    ----------
    items_gdf : geopandas.GeoDataFrame
        GeoDataFrame of STAC items to process.
    geometry : shapely.geometry.base.BaseGeometry
        Area of interest geometry in the target CRS.
    crs : pyproj.CRS, str or int, default=4326
        Coordinate reference system for the output.
    bands : list of str or int, optional
        List of band names or indices to select; if None, all valid bands are loaded.
    target_resolution : float or int, optional
        Target pixel size for the output raster.
    all_touched : bool, default=False
        Whether to include all pixels touched by the geometry.
    merge_method : {'last', 'first', 'min', 'max', 'mean', 'sum', 'median'}, default='last'
        Method to use when merging arrays.
    resampling_method : rasterio.enums.Resampling or str, default='nearest'
        Resampling method to use for reprojection.
    chunks : dict, optional
        Chunking options for dask/xarray.
    time_col : str, default='properties.datetime'
        Column name for datetime in items_gdf.
    time_format_str : str, optional
        Format string for parsing datetime values.
    ignore_time_component : bool, default=True
        If True, ignore the time component and only use the date.
    max_workers : int, default=1
        Number of parallel workers to use (-1 uses all available CPUs).
    enable_progress_bar : bool, default=True
        Whether to display a progress bar during processing.
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        The prepared time series raster data as an xarray DataArray with a time 
        dimension.
    """
    
    # need to handle case where chunks is passed, may not contain all dimensions
    # this is not strictly necessary, but it helps ensure that results are consistent
    # and expected as several functions will not assume dimensions like 'x', 'y', or 'time' are present
    passed_chunks = chunks.copy() if chunks is not None else None
    chunks_no_time = None
    if chunks is not None and isinstance(chunks, Dict):
        if 'x' not in chunks:
            chunks['x'] = -1
        if 'y' not in chunks:
            chunks['y'] = -1
        
        chunks_no_time = {k: v for k, v in chunks.items() if k != 'time'}
        if all([isinstance(b, str) for b in bands]):
            # as bands are stored in different files, need to remove the band
            # from chunk parameters when reading (can't chunk by band when only 
            # reading a file with a single band)
            if 'band' in chunks_no_time:
                chunks_no_time.pop('band')
    
    elif chunks is not None and isinstance(chunks, str):
        # if chunks is a string, we assume it is a dask chunking string like 'auto'
        chunks_no_time = chunks
    
    if time_col not in items_gdf.columns:
        raise ValueError(f"Column '{time_col}' not found in items_gdf. Please provide a valid time column.")
    items_gdf = items_gdf.sort_values(by=time_col)
    
    if ignore_time_component:
        items_gdf[time_col] = items_gdf[time_col].dt.normalize()
    
    if max_workers == 1:
        das = []
        for _, group_gdf in tqdm(
            items_gdf.groupby(time_col),
            desc="Processing items" if chunks_no_time is None else "Constructing dask computation graph",
            unit="timestep",
            disable=not enable_progress_bar,
        ):
            da = prepare_data(
                items_gdf=group_gdf,
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
                enable_progress_bar=False,
                chunks=chunks_no_time,
                **rioxarray_kwargs,
            )
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
            enable_time_dim=True,
            time_col=time_col,
            time_format_str=time_format_str,
            enable_progress_bar=False,
            chunks=chunks_no_time,
            **rioxarray_kwargs,           
        )
        
        groups = list(items_gdf.groupby(time_col))
        
        # Handle exceptions by wrapping the worker function
        def safe_worker(items_gdf):
            try:
                return worker(items_gdf=items_gdf)
            except Exception as e:
                warn(f"Failed to process a group with error: {e}. Skipping this group.")
                return None
        
        # Alternative approach with better error handling
        das = Parallel(n_jobs=max_workers)(
            delayed(safe_worker)(gpd.GeoDataFrame(group_gdf))
            for _, group_gdf in tqdm(
                groups,
                desc="Processing items" if chunks_no_time is None else "Constructing dask computation graph",
                unit="timestep",
                disable=not enable_progress_bar,
            )
        )
        das = [da for da in das if da is not None]
    
    da = xr.concat(das, dim='time').sortby('time')

    if passed_chunks is not None and isinstance(passed_chunks, dict):
        # Only compare specified dimensions
        needs_rechunk = False
        for dim, target in passed_chunks.items():
            current = da.chunksizes.get(dim)
            # If -1, treat as "all in one chunk"
            if target == -1:
                target = da.sizes[dim]
            if current is None or current[0] != target:
                needs_rechunk = True
                break
        if needs_rechunk:
            da = da.chunk(passed_chunks)
    
    elif passed_chunks is not None and isinstance(passed_chunks, str):
        da = da.chunk(passed_chunks)
    
    return da