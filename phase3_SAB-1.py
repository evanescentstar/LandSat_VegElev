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


def crsproj(shpcrs, newcrs):
    import pyproj
    shp = pyproj.CRS(f'EPSG:{shpcrs}')
    # conedcrs = conedobj.rio.crs.to_epsg()
    new1 = pyproj.CRS(f'EPSG:{newcrs}')
    project = pyproj.Transformer.from_crs(shp, new1, always_xy=True).transform
    return project


def shpplt(shp, crs1=None, notrans=True, **kwargs):
    import geopandas as gpd
    from cartopy import crs
    shpp = gpd.GeoSeries(shp)
    if crs1 == 'cpc':
        cpc = crs.PlateCarree()
        if not notrans:
            t1 = crsproj('3857', '4326')
            shp1a = transform(t1, shp)
            shp1ap = gpd.GeoSeries(shp1a)
        else:
            shp1ap = gpd.GeoSeries(shp)
        fig1 = plt.figure(figsize=(7, 6.5))
        ax0 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), projection=cpc, aspect='equal')
        gl = ax0.gridlines(crs=cpc, draw_labels=True,
                           linewidth=2, color='gray', alpha=0, linestyle='--')
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        shp1 = shp1ap.plot(ax=ax0, transform=cpc, **kwargs)
    else:
        fig1 = plt.figure(figsize=(7, 6.5))
        ax0 = fig1.add_axes((0.1, 0.1, 0.8, 0.8), aspect='equal')
        shp1 = shpp.plot(ax=ax0, **kwargs)
    return shp1


if 'd8' not in locals():
    with open('sab_flist.txt', 'r') as f1:
        flist = f1.readlines()
    f1.close()
    dcomp = rio.open_rasterio('nv/ATL_composite_veg_cmp.tif')
    d8 = rio.open_rasterio('nv/L8_2018_ATL_Unveg_Veg_Water_UVVR.tif')

# # plotting and selecting the range of the output tif
# llx0 = dcomp.x[0]
# lly0 = dcomp.y[-1]
# urx0 = dcomp.x[-1]
# ury0 = dcomp.y[0]
# p0 = cdg.poly_region((lly0,llx0,ury0,urx0))
# ax0 = shpplt(p0,'cpc')
# ax0.coastlines()
#
# outer_lx0 = -80.
# outer_ux0 = -80.
# outer_ly0 = 32.6
# outer_uy0 = 32.6
#
# for f1 in flist:
#     ds1 = rio.open_rasterio(f1.strip())
#     prj2 = crsproj(ds1.rio.crs.to_epsg(), dcomp.rio.crs.to_epsg())
#     llx1 = ds1.x[0]
#     lly1 = ds1.y[-1]
#     urx1 = ds1.x[-1]
#     ury1 = ds1.y[0]
#     p1 = cdg.poly_region((llx1, lly1, urx1, ury1), False)
#     p2 = transform(prj2, p1)
#     llx2, lly2, urx2, ury2 = p2.bounds
#     if llx2 < outer_lx0:
#         outer_lx0 = llx2
#     if lly2 < outer_ly0:
#         outer_ly0 = lly2
#     if urx2 > outer_ux0:
#         outer_ux0 = urx2
#     if ury2 > outer_uy0:
#         outer_uy0 = ury2
##  the numbers for these total SAB boundaries are:
##  -83.22727522918032, 27.403358914903787, -77.21155627371897, 35.24707769078961
## which informs the following selection of the total ATL composite region as follows
dcomp1 = dcomp[0, 36825:65935, :19089]

# collect GTIFF info to build new output file
dx, dy = dcomp1.rio.resolution()

gt2 = (float(dcomp1.x[0] - dx / 2), dx, 0, float(dcomp1.y[0] - dy / 2), 0, dy)

## create output file, of the same size
from osgeo import gdal

driver = gdal.GetDriverByName("GTiff")
# outdata = driver.Create('nv/test.tif', int(dcomp1.x.size), int(dcomp1.y.size), 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata = driver.Create('nv/newtest.tif', dcomp1.x.size, dcomp1.y.size, 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata.SetGeoTransform(gt2)
outdata.SetProjection(dcomp1.spatial_ref.attrs['spatial_ref'])
outdata.SetMetadata({'AREA_OR_POINT': 'Area'})

# initialize, in a fashion, with nan
band = outdata.GetRasterBand(1)
band.SetNoDataValue(-inf)
band.SetMetadata(['RepresentationType=ATHEMATIC'])
band.Fill(nan)
outdata.FlushCache()

### within file loop
for f1 in flist:
    ###f1 = flist[0]
    ds1 = rio.open_rasterio(f1.strip())
    prj1 = crsproj(dcomp.rio.crs.to_epsg(), ds1.rio.crs.to_epsg())
    prj2 = crsproj(ds1.rio.crs.to_epsg(), dcomp.rio.crs.to_epsg())
    llx1 = ds1.x[0]
    lly1 = ds1.y[-1]
    urx1 = ds1.x[-1]
    ury1 = ds1.y[0]

    p1 = cdg.poly_region((llx1, lly1, urx1, ury1), False)
    p2 = transform(prj2, p1)
    llx2, lly2, urx2, ury2 = p2.bounds

    if llx2 < dcomp1.x[0]:
        iflx = 0
    else:
        iflx = np.where(dcomp1.x < llx2)[0][-1]

    if lly2 < dcomp1.y[-1]:
        ifly = dcomp1.y.size - 1
    else:
        ifly = np.where(dcomp1.y < lly2)[0][0]

    if urx2 > dcomp1.x[-1]:
        icex = dcomp1.x.size - 1
    else:
        icex = np.where(dcomp1.x > urx2)[0][0]

    if ury2 > dcomp1.y[0]:
        icey = 0
    else:
        icey = np.where(dcomp1.y > ury2)[0][-1]

    dcomp2 = dcomp1[icey:ifly, iflx:icex]

    idx2 = np.where(dcomp2 > -1)

    icount = 1
    print(time.time())
    print(f'Starting the loop for valid landsat pixels in {f1}')
    sys.stdout.flush()

    for ii in range(len(idx2[0])):
        cur1 = dcomp2[idx2[0][ii], idx2[1][ii]]
        llx2a = cur1.x.values - dx / 2
        lly2a = cur1.y.values + dy / 2
        urx2a = cur1.x.values + dx / 2
        ury2a = cur1.y.values - dy / 2
        p2a = cdg.poly_region((llx2a, lly2a, urx2a, ury2a), False)
        p1a = transform(prj1, p2a)
        llx1a, lly1a, urx1a, ury1a = p1a.bounds

        lx = 5
        ux = -5
        ly = -5
        uy = 5
        if llx1a < ds1.x[0]:
            iflxa = 0
            lx = None
        else:
            iflxa = np.where(ds1.x < llx1a)[0][-1]

        if lly1a < ds1.y[-1]:
            iflya = ds1.y.size - 1
            ly = None
        else:
            iflya = np.where(ds1.y < lly1a)[0][0]

        if urx1a > ds1.x[-1]:
            icexa = ds1.x.size - 1
            ux = None
        else:
            icexa = np.where(ds1.x > urx1a)[0][0]

        if ury1a > ds1.y[0]:
            iceya = 0
            uy = None
        else:
            iceya = np.where(ds1.y > ury1a)[0][-1]

        cur2 = ds1[0, iceya:iflya + 1, iflxa:icexa + 1]
        cur2m = np.ma.masked_where(cur2 == cur2.attrs['_FillValue'], cur2)
        if cur2m.count() == 0:
            continue
        cur2m = cur2m / 1000.
        cur2m_1 = cur2m.repeat(10, axis=1).repeat(10, axis=0)

        # since the values are at midpoints of the 10-meter square cells,
        # must take 5 meters off the beginning and end of this array in
        # both directions, UNLESS one or more sides is on an edge of the file data
        cur2m_1 = cur2m_1[uy:ly, lx:ux]

        # find the offsets of the landsat pixel in this space from the
        # constructed finer mesh of the CoNED-based product
        iflxb = llx1a - cur2.x[0]
        if iflxb < 0:
            iflxb = 0
        else:
            iflxb = int(np.round(iflxb))
        icexb = cur2.x[-1] - urx1a
        if icexb < 0:
            icexb = None
        else:
            icexb = int(np.round(icexb * -1))
        iflyb = lly1a - cur2.y[-1]
        if iflyb < 0:
            iflyb = None
        else:
            iflyb = int(np.round(iflyb * -1))
        iceyb = cur2.y[0] - ury1a
        if iceyb < 0:
            iceyb = 0
        else:
            iceyb = int(np.round(iceyb))
        result1 = cur2m_1[iceyb:iflyb, iflxb:icexb]
        if result1.mask.all():
            continue
        if cur1.values == 1.:
            result1a = float(result1.mean())
        elif cur1.values == 0.:
            continue
        else:
            result1 = result1.flatten()[~result1.flatten().mask]
            if result1.size == 0:
                continue
            result1.sort()
            # veg fraction of number of points in CoNED for shape
            vf_conpts = result1.shape[0] * cur1.values

            # mean of highest 'vegetated fraction' of sorted CoNED elevations
            result1a = float(np.mean(result1[-int(vf_conpts):]))
        ## write to output file, remember, x first!  Only flush every thousand writes.
        band.WriteArray(np.array([[result1a]]), int(iflx+idx2[1][ii]), int(icey+idx2[0][ii]))
        icount = icount + 1
        if (icount % 1000) == 0:
            outdata.FlushCache()
            print('wrote another 1000\n')
            sys.stdout.flush()

    print(time.time())

    print(f'Finished {f1}')
    sys.stdout.flush()

