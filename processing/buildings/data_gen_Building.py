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
import sys
# Get the parent directory
sys.path.append('..')
from tools.building_data_utils import *


def create_meshgrid(gdf_, step=10, kw='geometry', plot=True):
    """
    Create a meshgrid of coordinates based on the bounds of a geodataframe
    Args:
        gdf_: geopandas dataframe
        step: int
        kw: str
        plot: bool
    Returns: list of tuples of coordinates (x, y) of type float
    """
    
    # Define the dimensions
    xmax, ymax = 99, 99

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(0, xmax, step), 
                         np.arange(0, ymax, step))

    # Flatten and collect x and y coordinates
    x = [z for x in xx for z in x]
    y = [z for x in yy for z in x]

    # Combine x and y coordinates into tuples
    coordinates = list(zip(x, y))

    # Shuffle the coordinates
    np.random.shuffle(coordinates)

    if plot:
        plt.plot(xx.flat, yy.flat, ".")
        plt.show()

    return coordinates


def create_random_grid(gdf_, n_points, kw='geometry', plot=True):
    """
    Create a random grid of coordinates based on the bounds of a geodataframe
    Args:
        gdf_: geopandas dataframe
        n_points: int
        kw: str
        plot: bool
    Returns: list of tuples of coordinates (x, y) of type float
    """
    # xmax = max(max([(poly.bounds[0], poly.bounds[2]) for poly in gdf_[kw]]))
    # ymax = max(max([(poly.bounds[1], poly.bounds[3]) for poly in gdf_[kw]]))
    xmax, ymax = 99, 99
    # Generate a list of all possible (x, y) coordinates
    all_possible_pins = [(x, y) for x in range(xmax) for y in range(ymax)]

    # Shuffle the list to randomize the order
    random.shuffle(all_possible_pins)

    # Select the first n coordinates to ensure uniqueness
    unique_pins = all_possible_pins[:n_points]
    
    if plot:
        xs, ys = zip(*unique_pins)
        plt.plot(xs, ys, ".")
        
    return unique_pins #list(zip(x, y))


def generate_pin_counts(data_list, rad=20, step=100, n_pins=100, stop=6, kw='geometry', off=(20, 20), gridtype="random", plot=True):
    """
    Generate random pins (x, y) and count of buildings 
    in a given list of geojsons
    Args:
        data_list: list of tuples of strings
        rad: int
        step: int
        stop: int
        kw: str
        off: tuple of ints
        plot: bool
    Returns: list of lists of strings and lists of tuples of floats and ints
    """
    out_ = []
    N = 0
    print(f"Using {gridtype} grid")
    for rastr, geojs in data_list:
        gdf_ = scale_and_translate((rastr, geojs), plot=plot)
        if gridtype == "random":
            grid = create_random_grid(gdf_, n_pins, plot=plot)
            
        elif gridtype == "mesh":           
            grid = create_meshgrid(gdf_, step=step, kw=kw, plot=plot)
        circs = [Polygon(circle_coords_fn(9*circ_coors[0], 9*circ_coors[1], rad, step, offset=off)) for circ_coors in grid]
        if plot: plot_gdf_circle(gdf_, circs, multicircs=True, kw=kw)
        
        cg = list(zip(circs, grid))
        lst = [((g[0], g[1]), len(geopandas.sjoin(geopandas.GeoDataFrame(index=[0], crs='epsg:32631', geometry=[c]), gdf_))) for c,g in cg]
        out_.append([os.path.basename(rastr), [loc for loc, _ in lst], [cnt for _, cnt in lst]])
        N += 1

        if N%100==0:
            print(f"{N} of {len(data_list)}")
        if N >= stop:
            break
    return out_


# change datapath here
sample_size = 1000
KW = 'PS-RGBNIR'
data = "../data/Building"

tifs_pth = f"{os.path.join(data, KW)}/*.tif"

geojsons_pth =f"{os.path.join(data, 'geojson_buildings')}/*.geojson"
print(tifs_pth, geojsons_pth)
geojsons_pth =f"{os.path.join(data, 'geojson_buildings')}/*.geojson"
print(f"total number of clean data:{len(remove_empty_geojsons(get_data(tifs_pth, geojsons_pth)))}")
print(f"total number of data:{len(get_data(tifs_pth, geojsons_pth))}")
# uncomment the line if you want first sample_size images
datas = remove_empty_geojsons(get_data(tifs_pth, geojsons_pth)[:sample_size])
#full dataset
# datas = remove_empty_geojsons(get_data(tifs_pth, geojsons_pth))


# copy tifs from AOI_11_Rotterdam to AOI_11_Rotterdam/processed/kw
if not os.path.exists(f"{os.path.join(data, 'processed', KW)}"):
    for tif, _ in datas:
        fname = os.path.basename(tif)
        save_dir = os.path.split(os.path.dirname(tif))[0]
        if not os.path.exists(f'{save_dir}/processed/{KW}'):
            os.makedirs(f'{save_dir}/processed/{KW}')
        output_file = f'{save_dir}/processed/{KW}/{fname}'
        if os.path.exists(output_file):
            print(f'{output_file} already exists')
            continue
        os.system(f'cp {tif} {output_file}')
else:
    print("tifs already copied")
    
    
step_list = [288, 90, 27]

for step in step_list:
    out = generate_pin_counts(datas, rad=30, step=step, stop=len(datas), kw='geometry', off=(0, 0), gridtype="mesh", plot=False)
    # pd.DataFrame(out).to_csv(f'building_{sample_size}_{KW}_{step}_{len(out[0][2])}_mesh.csv')
    pd.DataFrame(out).to_csv(f'building_mesh_{step}.csv', index=False)
    print(len(out[0][2]))
    
    
n_pins_list = [10, 100, 200] #, 100, 200
testgdf = scale_and_translate(datas[0], plot=False)
for n_pins in n_pins_list:
    psrgbnir_grid = create_random_grid(testgdf, n_pins, kw='geometry', plot=True)
    save_array(psrgbnir_grid, f'psrgbnir_grid_{n_pins}.npy', overwrite=True)

    start = time.time()
    step = 200
    out = generate_pin_counts(datas, rad=30, n_pins=n_pins, stop=len(datas), kw='geometry', off=(0, 0), gridtype="random", plot=False)
    pd.DataFrame(out).to_csv(f'Building_{n_pins}_random.csv', index=False)
    print(f"--- {time.time() - start} seconds ---")
    print(len(out[0][2]))