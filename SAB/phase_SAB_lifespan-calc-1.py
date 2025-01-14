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
import scipy as sp
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

def lspan(UVVR, A_px, Ext_inorg, Veg_org, Veg_elev, A_reveg, V_reveg, Bulk_exist, Slr_bgrnd, Slr_fut, Bulk_fut):
    A_veg = A_px / (1 + UVVR)
    Sed_budget = (-0.416 * np.log(np.max([UVVR, 0.0001])) - 1.0749) * A_px + Ext_inorg + Veg_org * A_veg
    Veg_elev_mod = (Veg_elev * (A_veg - A_reveg) + V_reveg) / A_veg
    Sed_tot = Veg_elev_mod * A_veg * Bulk_exist
    Sed_budget_fut = Sed_budget - ((Slr_fut - Slr_bgrnd) * Bulk_fut) * A_px
    if Sed_budget_fut > 0:
        return None
    Lifespan = -1 * Sed_tot / Sed_budget_fut
    return Lifespan

duvvr = rio.open_rasterio('nv/ATL_composite_uvvr_cmp.tif')
duvvr1 = duvvr[0, 36825:65935, :19089]

delv1 = rio.open_rasterio('sab_fin3a.tif')


# collect GTIFF info to build new output file
dx, dy = delv1.rio.resolution()

gt2 = (float(delv1.x[0] - dx / 2), dx, 0, float(delv1.y[0] - dy / 2), 0, dy)

## create output file, of the same size
from osgeo import gdal

driver = gdal.GetDriverByName("GTiff")
# outdata = driver.Create('nv/test.tif', int(delv1.x.size), int(delv1.y.size), 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata = driver.Create('lifespan-1.tif', delv1.x.size, delv1.y.size, 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata.SetGeoTransform(gt2)
outdata.SetProjection(delv1.spatial_ref.attrs['spatial_ref'])
outdata.SetMetadata({'AREA_OR_POINT': 'Area'})

# initialize, in a fashion, with nan
band = outdata.GetRasterBand(1)
band.SetNoDataValue(-inf)
band.SetMetadata(['RepresentationType=ATHEMATIC'])
band.Fill(nan)
outdata.FlushCache()


ind1 = np.where(~delv1[0].isnull())

A_px_eq = (dx*111000)**2

icount = 0
st = time.time()
for ii in range(len(ind1[0])):
    # lspan(UVVR, A_px, Ext_inorg, Veg_org, Veg_elev, A_reveg, V_reveg, Bulk_exist, Slr_bgrnd, Slr_fut, Bulk_fut)
    cur_uvvr = float(duvvr1[ind1[0][ii],ind1[1][ii]].values)
    cur_elv1 = float(delv1[0,ind1[0][ii],ind1[1][ii]].values)
    lat1 = float(delv1[0,ind1[0][ii],ind1[1][ii]].y)
    latfac = sp.special.cosdg(lat1)
    A_px = A_px_eq*latfac
    lspan1 = lspan(cur_uvvr, A_px, 0, 0, cur_elv1, 0, 0, 373, 0.00348, 0.007, 159)
    if lspan1 is not None:
        pass
        band.WriteArray(np.array([[lspan1]]), int(ind1[1][ii]), int(ind1[0][ii]))
    icount = icount + 1
    if (icount % 10000) == 0:
        t1 = time.time() - st
        print(f'icount is {icount} :: time was {t1} seconds')
        sys.stdout.flush()
        # every 10000, write out the cached writes
        if lspan1 is not None:
            pass
            outdata.FlushCache()
        st = time.time()


