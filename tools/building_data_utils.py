import os
from os.path import dirname as up
import math
import glob
import random
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window, get_data_window
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon, MultiPolygon
import pyproj
import shapely.ops
from shapely.geometry import shape
from geopandas import GeoSeries
import time
random.seed(10)

def plot_sat(path):
    """
    Plot satellite image with optional geodataframe overlay
    Args:
        path: str
        gdf_: geopandas dataframe
        linewidth: int
    """
    f, ax = plt.subplots(1,figsize=(3, 3))
    f.tight_layout()
    
    r = rio.open(path)
    r = r.read()
    r = r.transpose((1,2,0,))
    
    ax.imshow(r, origin='lower')
    
    
def plot_gdf(gdf_, ax=None, linewidth=2, kw='geometry'):
    """
    Plot geodataframe
    Args:
        gdf_: geopandas dataframe
        ax: matplotlib axis
        linewidth: int
        kw: str
    Returns: matplotlib axis
    """
    if ax is None:
        _,ax = plt.subplots(1,figsize=(3, 3))
        
    for geom in gdf_[kw]:
        try:
            ax.plot(*geom.exterior.xy,linewidth=linewidth)
        except AttributeError:
            # MultiPolygon
            for poly in geom.geoms:
                ax.plot(*poly.exterior.xy,linewidth=linewidth)
    return ax


def plot_gdf_circle(gdf_, circs, ax=None, linewidth=2, multicircs=True, kw='geometry'):
    """
    Plot geodataframe with circles
    Args:
        gdf_: geopandas dataframe
        circs: list of shapely polygons
        ax: matplotlib axis
        linewidth: int
        multicircs: bool
        kw: str
    Returns: matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(1,figsize=(3, 3))
    
    for geom in gdf_[kw]:
        try:
            ax.plot(*geom.exterior.xy,linewidth=linewidth)
        except AttributeError:
            # MultiPolygon
            for poly in geom.geoms:
                ax.plot(*poly.exterior.xy,linewidth=linewidth)
    if multicircs:
        for circ_ in circs:
            ax.plot(*circ_.exterior.xy, linewidth=linewidth)
            ax.axis('tight')
    else: ax.plot(*circs.exterior.xy, linewidth=linewidth)
    
    return ax


def get_circle_coord(theta, x_center, y_center, radius):
    """
    Get the coordinates of a point on a circle given the angle, center, and radius
    Args:
        theta: float
        x_center: float
        y_center: float
        radius: int
    Returns: tuple of floats
    """
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return x, y

def get_all_circle_coords(x_center, y_center, radius, n_points):
    """
    Get all the coordinates of a circle given the center, radius, and number of points
    Args:
        x_center: float
        y_center: float
        radius: int
        n_points: int
    Returns: list of tuples of floats
    """
    thetas = [i/n_points * math.tau for i in range(n_points)]
    circle_coords_ = [get_circle_coord(theta, x_center, y_center, radius) for theta in thetas]
    return circle_coords_

def circle_coords_fn(x_center, y_center, radius, n_points, offset=(20,20)):
    """
    Compute circle coordinates using a center point (x, y)
    and radius. The number of coordinates computed is n_points.
    Args:
        x_center: float
        y_center: float
        radius: int
        n_points: int
        offset: tuple of ints  
    Returns: list of tuples of floats
    """
    thetas = [i/n_points * math.tau for i in range(n_points)]
    xoff, yoff = offset
    return [(radius*math.cos(theta)+(x_center-xoff), radius*math.sin(theta)+(y_center-yoff)) for theta in thetas]

# circle_coords = circle_coords_fn(x_center = 595134.0, 
#                                       y_center = 5751614.0,
#                                       radius = 30,
#                                       n_points = 100)


def get_data(tifs_path, geojsons_path):
    """
    Process tifs and geojsons to be used for pin generation
    Args:
        tifs_path: str
        geojsons_path: str   
    Returns: list of tuples of strings
    """
    tifs = sorted(glob.glob(tifs_path), key=lambda s: int(s.split("_tile_")[1].split('.')[0]))
    geojsons = sorted(glob.glob(geojsons_path), key=lambda s: int(s.split("_tile_")[1].split('.')[0]))
    geojsons = [g for g in geojsons if g.split("_tile_")[1].split('.')[0] \
                in [t.split("_tile_")[1].split('.')[0] for t in tifs]]
    datas_ = list(zip(tifs, geojsons))
    assert len(tifs) == len(geojsons), "tifs and geojsons must be of equal length"
    assert len(datas_) > 0, "no data found"
    return datas_


def remove_empty_geojsons(data_):
    """
    Remove empty geojsons from the data
    Args:
        data_: list of tuples of strings 
    Returns: list of tuples of strings
    """
    out_ = []
    for tif, geojson in data_:
        gdf_ = geopandas.read_file(geojson)
        if len(gdf_['geometry']) > 0:
            out_.append((tif, geojson))
    return out_


def scale_and_translate(data_, plot=True):
    """
    Scale and translate the geojsons to pixel coordinates
    Args:
        data_: tuple of strings
        plot: bool
    Returns: geopandas dataframe
    """
    # data_ is a tuple of (tif, geojson)
    tif = data_[0]
    gdf_ = geopandas.read_file(data_[1])
    gdf_['px_geometry'] = 0
    column_index = gdf_.columns.get_loc('geometry')
    
    src_ = rio.open(tif)
    gdf_['px_geometry'] = gdf_['geometry'].apply(lambda x: rio.transform.rowcol(src_.transform, [i[0] for i in x.exterior.coords], [i[1] for i in x.exterior.coords]) \
        if x.geom_type == 'Polygon' else [rio.transform.rowcol(src_.transform, [i[0] for i in poly.exterior.coords], [i[1] for i in poly.exterior.coords]) for poly in list(x.geoms)])

    # make the pixel geometry column a shapely polygon
    for coords in enumerate(gdf_['px_geometry']):
        try:
            gdf_.iloc[coords[0], column_index] = Polygon(list(zip(coords[1][0], coords[1][1])))
        except TypeError:
            polys = []
            for poly in coords[1]:
                polys.append(Polygon(list(zip(poly[0], poly[1]))))
            gdf_.iloc[coords[0], column_index] = MultiPolygon(polys)

    miny = min([poly.bounds[1] for poly in gdf_['geometry']])
    gdf_['tr_geometry'] = GeoSeries(gdf_['geometry']).translate(xoff=0, yoff=np.abs(miny))

    # rotate the pixel geometry
    gdf_['ro_geometry'] = GeoSeries(gdf_['tr_geometry']).rotate(270, origin=(0, 0))

    miny = min([poly.bounds[1] for poly in gdf_['ro_geometry']])
    gdf_['tr2_geometry'] = GeoSeries(gdf_['ro_geometry']).translate(xoff=0, yoff=np.abs(miny))
    
    if plot: plot_gdf(gdf_, linewidth=2, kw='geometry')
    return gdf_


def crop_tifs_to_geojsons(data_, kw, ckw, overwrite=False):
    """
    Crop tifs to the bounds of the geojsons
    Args:
        data_: list of tuples of strings
        kw: str
        overwrite: bool  
    Returns: list of tuples of strings
    """
    dat = []
    for tif, gjson in data_:
        input_file = tif
        fname = os.path.basename(input_file)
        save_dir = os.path.split(os.path.dirname(input_file))[0]
        if not os.path.exists(f'{save_dir}/processed/cropped/{kw}'):
            os.makedirs(f'{save_dir}/processed/cropped/{kw}')
        output_file = f'{save_dir}/processed/cropped/{kw}/{fname}'
        dat.append((output_file, gjson))
        if os.path.exists(output_file) and not overwrite:
            print(f'{output_file} already exists')
            continue
        gdf_ = scale_and_translate((tif, gjson), plot=False)
        x_max = max([poly.bounds[0] for poly in gdf_[ckw]])
        y_max = max([poly.bounds[2] for poly in gdf_[ckw]])
        with rio.open(input_file, 'r+') as src_:
            profile = src_.profile
            window = Window(0, 0, x_max, y_max)
            data_ = src_.read(window=window)
            profile.update({
                'driver': src_.driver,
                'height': window.height,
                'width': window.width,
                'transform': rio.windows.transform(window, src_.transform)
            })
            with rio.open(
                    output_file,
                    'w',
                    **profile
            ) as dst:
                dst.write(data_)
    return dat


def mask_and_rotate_data(data_, kw, rotate=False, overwrite=False):
    """
    Mask and rotate the data
    Args:
        data_: list of tuples of strings
        kw: str
        rotate: bool
        overwrite: bool
    Returns: list of tuples of strings
    """
    dat = []
    for tif, gjson in data_:
        input_file = tif
        fname = os.path.basename(input_file)
        save_dir = os.path.split(up(up(up(input_file))))[0]
        # use the same kw as the tifs modality
        if not os.path.exists(f'{save_dir}/processed/masked/{kw}'):
            os.makedirs(f'{save_dir}/processed/masked/{kw}')
        output_file = f'{save_dir}/processed/masked/{kw}/{fname}'
        dat.append((output_file, gjson))
        if os.path.exists(output_file) and not overwrite:
            print(f'{output_file} already exists')
            continue
        with rio.open(input_file, 'r+') as src_:
            src_.nodata = 0
            profile = src_.profile.copy()
            window = get_data_window(src_.read(masked=True))
            window_data = src_.read(window=window)
            if rotate:
                final_data = np.rot90(window_data, k=1, axes=(1, 2))
                transform = src_.window_transform(window) * rio.Affine.rotation(-90)
            else:
                final_data = window_data
                transform = src_.window_transform(window)
            profile.update(
                driver=src_.driver,
                transform=transform,
                height=window.height,
                width=window.width)
            with rio.open(
                    output_file,
                    'w',
                    **profile
            ) as dst:
                dst.write(final_data)
    return dat


# save array as .npy
def save_array(array, name, overwrite=False):
    """
    Save array as .npy
    Args:
        array: np.array
        name: str
    """
    if os.path.isfile(name) and not overwrite:
        print(f'{name} already exists')
    else:
        np.save(name, array)