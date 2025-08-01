from time import sleep
from typing import Optional
from warnings import warn
import numpy as np
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from pyproj import CRS
import planetary_computer
from typing import List, Dict, Any, Union
import odc.geo.xr
from odc.geo.geom import Geometry
import os
from typing import Union
import xarray as xr
from pandas import notna



def load_from_url(
    url: str,
    bands: Optional[List[int]] = None,
    geometry: Optional[BaseGeometry] = None,
    crs: Union[CRS, str, int] = 4326,
    chunks: Optional[Dict[str, int]] = None,
    clip_to_geometry: bool = False,
    all_touched: bool = False,
    max_retries: int = 5,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    """
    Load a raster dataset from a URL and return it as an xarray DataArray.

    Signs the provided URL using the Planetary Computer, opens the raster using 
    rioxarray, optionally selects bands, and clips to a geometry if provided. The 
    raster is returned in its original CRS, but can be clipped using the provided 
    CRS.

    Parameters
    ----------
    url : str
        The URL of the raster dataset to load (will be signed if needed).
    bands : list of int, optional
        List of band indices to select. If None, all bands are loaded.
    geometry : shapely.geometry.base.BaseGeometry, optional
        Geometry to clip the raster data to. If provided, the raster is clipped 
        to this geometry.
    crs : pyproj.CRS, str or int, default=4326
        Coordinate reference system for clipping. Does not reproject the raster.
    chunks : dict, optional
        Chunking options for dask/xarray.
    clip_to_geometry : bool, default=False
        If True, apply a mask to the raster using the provided geometry.
    all_touched : bool, default=False
        Whether to include all pixels touched by the geometry during clipping.
    max_retries : int, default=5
        Maximum number of attempts to load the raster in case of failure.
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments passed to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        The loaded (and optionally clipped) raster data.

    Raises
    ------
    RuntimeError
        If the raster cannot be loaded after the specified number of retries.
    """
    
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"  # Disable directory reading for performance
    os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = '10'   # connection-level timeout
    os.environ["GDAL_HTTP_TIMEOUT"] = "10"          # request-level timeout
    os.environ["GDAL_HTTP_MAX_RETRY"] = "0"         # number of retry attempts
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "0"       # seconds between retries

    
    for _retries in range(max_retries):
        try:
            # Sign the URL with Planetary Computer
            signed_url = planetary_computer.sign(url)
            da = rxr.open_rasterio(
                signed_url,
                chunks=chunks,
                masked=True,
                lock=False,
                **rioxarray_kwargs
            )

            break  # Exit loop if loading is successful
        except Exception as e:
            if _retries < max_retries - 1:
                warn(f"Retrying loading raster from COG to error: {e}. Attempt {_retries + 1}/{max_retries}")
                sleep(2 ** _retries)  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to load raster from COG after {max_retries} attempts: {e}")
    
    if bands is not None:
        da = da.sel(band=bands)
    
    if geometry is not None: 
        da = da.odc.crop(Geometry(geometry, crs=crs), apply_mask=clip_to_geometry, all_touched=all_touched)
        
    return da



def read_single_item(
    item_gs: gpd.GeoSeries,
    bands: Optional[List[Union[str, int]]] = None,
    geometry: Optional[BaseGeometry] = None,
    crs: Union[CRS, str, int] = 4326,
    chunks: Optional[Dict[str, int]] = None,
    clip_to_geometry: bool = True,
    all_touched: bool = False,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    """
    Read a single STAC item into an xarray DataArray, selecting and concatenating bands as needed.

    Identifies the appropriate asset URLs from a STAC item (GeoSeries), loads each 
    band as a DataArray, reprojects/resamples as needed, and concatenates them along 
    the 'band' dimension. If only one band is selected, returns a single DataArray.

    Parameters
    ----------
    item_gs : geopandas.GeoSeries
        A STAC item record with asset hrefs and metadata.
    bands : list of str or int, optional
        Band names or indices to select. If strings, must match asset keys. If None, 
        all valid bands are loaded.
    geometry : shapely.geometry.base.BaseGeometry, optional
        Geometry to clip the raster data to. If provided, the raster is clipped 
        to this geometry.
    crs : pyproj.CRS, str or int, default=4326
        Output coordinate reference system for clipping. Does not reproject the 
        raster.
    chunks : dict, optional
        Chunking options for dask/xarray.
    clip_to_geometry : bool, default=True
        If True, apply a mask to the raster using the provided geometry.
    all_touched : bool, default=False
        Whether to include all pixels touched by the geometry during clipping.
    **rioxarray_kwargs : dict, optional
        Additional keyword arguments passed to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        If only one band is selected, returns a DataArray. If multiple bands, returns 
        a concatenated DataArray along the 'band' dimension.
    """
    
    ignored_assets = [
        'assets.visual.href', # RGB composite found in S2-l2a collections
        'assets.thumbnail.href', # Thumbnail found in S2-l2a collections
        'assets.preview.href', # Preview image found in S2-l2a collections
    ]
    
    if bands is not None and all(isinstance(band, str) for band in bands):
        selected_columns = [f'assets.{band}.href' for band in bands if f'assets.{band}.href' in item_gs.index]
        missing_bands = [band for band in bands if f'assets.{band}.href' not in item_gs.index]
        if missing_bands:
            warn(f"Some requested bands are not available in the item: {missing_bands}. Available bands will be loaded instead.")
    else:
        selected_columns = [
            col for col in item_gs.index \
            if col.startswith('assets.') \
            and col.endswith('.href') \
            and notna(item_gs[col.replace('.href', '.type')]) \
            and 'image/tiff; application=geotiff; profile=cloud-optimized' in item_gs[col.replace('.href', '.type')] \
            and col not in ignored_assets
        ]
    
    if len(selected_columns) == 1:
        # if only one column is selected, we can read it directly
        return load_from_url(
            url=item_gs[selected_columns[0]],
            bands=bands if bands is not None and all([isinstance(band, int) for band in bands]) else None,
            geometry=geometry,
            crs=crs,
            chunks=chunks,
            clip_to_geometry=clip_to_geometry,
            all_touched=all_touched,
            **rioxarray_kwargs
        )
        
    # Need to handle the case where a user does not specify chunking along the 'band' dimension
    # else, each band will effectively comprise 1 chunk, which may not be ideal
    if chunks is not None and isinstance(chunks, Dict) and 'band' in chunks:
        warn_str = "Chunking along the 'band' dimension is not supported when reading from multiple COG URLs " + \
            "(i.e., when bands are loaded from separate files). " + \
            "This can cause many expensive rechunk operations in the Dask computation graph. " + \
            "Only set chunking for spatial or time dimensions in this case."  + \
            "Removing 'band' chunking from the provided chunks."
        warn(warn_str)
        chunks = {k: v for k, v in chunks.items() if k != 'band'}
    
    band_das = []
    for col in selected_columns:
        url = item_gs[col]
        band_name = col.split('.')[-2]
        band_da = load_from_url(
            url=url,
            geometry=geometry,
            crs=crs,
            chunks=chunks,
            clip_to_geometry=clip_to_geometry,
            all_touched=all_touched,
            **rioxarray_kwargs,
        )

        if 'band' not in band_da.dims:
            band_da = band_da.expand_dims(band=[band_name])
        elif band_da.band.values[0] != band_name:
            band_da = band_da.assign_coords(band=[band_name])
        band_das.append(band_da)
    
    # Resample bands to the smallest pixel resolution
    target_band = min(band_das, key=lambda x: min(abs(x.rio.resolution()[0]), abs(x.rio.resolution()[1])))
    target_geobox = target_band.odc.geobox
    
    # nodata value selection logic
    nodata_values = [da.rio.nodata for da in band_das if da.rio.nodata is not None]
    # check if all nodata values are NaN
    nodata_values_arr = np.array(nodata_values)
    if np.all(np.isnan(nodata_values_arr)):
        num_unique_nd_values = 1
    else:
        num_unique_nd_values = len(set(nodata_values_arr[~np.isnan(nodata_values_arr)])) + int(np.any(np.isnan(nodata_values_arr)))
    
    nodata_value = band_das[0].rio.nodata
    if num_unique_nd_values > 1:
        warn(f"Multiple nodata values found across bands({nodata_values}). Using the first band's nodata value ({nodata_value}).")
    
    # Reproject bands that don't match the target band's spatial properties
    reprojected_bands = []
    for da in band_das:
        if da.odc.geobox == target_geobox:
            # No reprojection needed
            reprojected_bands.append(da)
        
        else: # Reproject to match the target geobox
            # for some reason nodata is changed wihen reprojecting
            # not sure if this is a bug in odc-geo or expected behavior
            reprojected_da = da.odc.reproject(
                target_geobox,
                resampling='nearest',  # Use nearest neighbor for categorical data
            )
            reprojected_da = reprojected_da.rio.write_nodata(nodata_value)
            reprojected_bands.append(reprojected_da)
    
    da = xr.concat(reprojected_bands, dim='band')
    da = da.rio.write_nodata(nodata_value)
    
    return da

