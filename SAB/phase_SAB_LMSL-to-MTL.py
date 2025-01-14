import rioxarray as rio
from shapely import geometry
from cartopy.io import shapereader as shpreader
from osgeo import osr
import geopandas as gpd
import coawst_dataget as cdg
from shapely.ops import transform
from rasterio.transform import AffineTransformer, Affine
from rasterio.features import rasterize
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Point, MultiPoint

def crsproj(shpcrs, newcrs):
    import pyproj
    shp = pyproj.CRS(f'EPSG:{shpcrs}')
    # conedcrs = conedobj.rio.crs.to_epsg()
    new1 = pyproj.CRS(f'EPSG:{newcrs}')
    project = pyproj.Transformer.from_crs(shp, new1, always_xy=True).transform
    return project

dcomp1 = rio.open_rasterio('../veg/sab_fin3.tif')
dmtl = rio.open_rasterio('SAB_LMSL_to_MTL.tif')
prj1 = crsproj(dcomp1.rio.crs.to_epsg(), dmtl.rio.crs.to_epsg())

ind1 = np.where(~dcomp1[0].isnull())

icount = 0
st = time.time()
for ii in range(len(ind1[0])):
    pt1 = transform(prj1,Point(dcomp1[0,ind1[0][ii],ind1[1][ii]].x,dcomp1[0,ind1[0][ii],ind1[1][ii]].y))
    mx, my = pt1.x, pt1.y
    ix = np.where(np.abs(dmtl.x - mx) <= 125)[0]
    if len(ix) > 1:
        iix = int(np.argsort(np.abs(dmtl.x - mx)[ix])[0])
        ix1 = ix[iix]
    elif len(ix) == 1:
        ix1 = ix[0]
    else:
        continue
    iy = np.where(np.abs(dmtl.y - my) <= 125)[0]
    if len(iy) > 1:
        iiy = int(np.argsort(np.abs(dmtl.y - my)[iy])[0])
        iy1 = iy[iiy]
    elif len(iy) == 1:
        iy1 = iy[0]
    else:
        continue
    if dmtl[0, iy1, ix1].isnull():
        continue
    del_mtl = float(dmtl[0, iy1, ix1].values)
    dcomp1[0,ind1[0][ii],ind1[1][ii]] = dcomp1[0,ind1[0][ii],ind1[1][ii]] - del_mtl
    icount = icount + 1
    if (icount % 10000) == 0:
        t1 = time.time() - st
        print(f'icount is {icount} :: time was {t1} seconds')
        sys.stdout.flush()
        st = time.time()






