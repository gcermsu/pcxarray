
from typing import Optional
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from pyproj import CRS
import planetary_computer
from typing import List, Dict, Any, Union


def load_from_url(
    url: str,
    geometry: Optional[BaseGeometry] = None,
    bands: Optional[List[int]] = None,
    crs: Union[CRS, str] = 4326,
    compute: bool = False,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> xr.DataArray:
    
    """
    Load a raster dataset from a URL, optionally clipping to a geometry and selecting bands.

    Parameters
    ----------
    url : str
        The URL of the raster dataset.
    masked : bool, optional
        Whether to mask the raster data (default is False).
    chunks : dict, optional
        Chunking options for dask/xarray (default is None).
    geometry : shapely.geometry.base.BaseGeometry, optional
        Geometry to clip the raster data to (default is None).
    bands : list of int, optional
        List of band indices to select (default is None, which selects all bands).
    crs : Union[CRS, str], optional
        Coordinate reference system for the output (default is 4326).
    compute: bool, optional
        Whether to compute the DataArray immediately (default is False, which returns a lazy DataArray
    **rioxarray_kwargs: dict, optional
        Additional keyword arguments to pass to `rioxarray.open_rasterio`.
    Returns
    -------
    xarray.DataArray
        The loaded raster data as an xarray DataArray.
    """
    
    signed_url = planetary_computer.sign(url)
    
    da = rxr.open_rasterio(signed_url, **rioxarray_kwargs)
    
    if bands is not None:
        da = da.sel(band=bands)
    
    if geometry is not None: # clip_box and sel are lazy operations, while clip is not
        da = da.rio.clip_box(*geometry.bounds, crs=crs)
        da = da.rio.clip([geometry], crs=crs, all_touched=True)

    if compute:
        return da.compute()
        
    return da



def read_single_item(
    item_gs: gpd.GeoSeries,
    geometry: Optional[BaseGeometry] = None,
    bands: Optional[List[Union[str, int]]] = None,
    crs: Union[CRS, str] = 4326,
    **rioxarray_kwargs: Optional[Dict[str, Any]]
) -> Union[xr.DataArray, tuple]:
    
    ignored_assets = [
        'assets.visual.href', # RGB composite found in S2-l2a collections
        'assets.thumbnail.href', # Thumbnail found in S2-l2a collections
        'assets.preview.href', # Preview image found in S2-l2a collections
    ]
    
    if bands is not None and all(isinstance(band, str) for band in bands):
        selected_columns = [f'assets.{band}.href' for band in bands if f'assets.{band}.href' in item_gs.index]
    else:
        selected_columns = [
            col for col in item_gs.index \
            if col.startswith('assets.') \
            and col.endswith('.href') \
            and 'image/tiff; application=geotiff; profile=cloud-optimized' in item_gs[col.replace('.href', '.type')] \
            and col not in ignored_assets
        ]
    
    
    if len(selected_columns) == 1:
        # if only one column is selected, we can read it directly
        return load_from_url(
            url=item_gs[selected_columns[0]],
            geometry=geometry,
            bands=bands if bands is not None and all(isinstance(band, int) for band in bands) else None,
            crs=crs,
            **rioxarray_kwargs
        )
    
    band_das = []

    for col in selected_columns:
        url = item_gs[col]
        band_name = col.split('.')[-2]
        band_da = load_from_url(
            url=url,
            geometry=geometry,
            crs=crs,
            **rioxarray_kwargs,
        )

        if 'band' not in band_da.dims:
            band_da = band_da.expand_dims(band=[band_name])
        elif band_da.band.values[0] != band_name:
            band_da = band_da.assign_coords(band=[band_name])
        band_das.append(band_da)
    
    # Resample bands to the smallest pixel resolution
    target_band = min(band_das, key=lambda x: min(abs(x.rio.resolution()[0]), abs(x.rio.resolution()[1])))
    
    # Reproject bands that don't match the target band's spatial properties
    reprojected_bands = []
    for da in band_das:
        needs_reprojection = (
            da.rio.crs != target_band.rio.crs or
            da.rio.resolution() != target_band.rio.resolution() or
            da.rio.bounds() != target_band.rio.bounds() or
            da.rio.transform() != target_band.rio.transform()
        )
        
        if needs_reprojection:
            reprojected_bands.append(da.rio.reproject_match(target_band))
        else:
            reprojected_bands.append(da)
    band_das = reprojected_bands
    
    da = xr.concat(band_das, dim='band')
    return da

