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
from matplotlib import pyplot as plt
import xarray as xr
from matplotlib import cm

logfile = 'p3-3a.log'
bndlogfile = 'p3-bnd-3a.log'  # log pixels with data bound problems (crosses boundary of CoNED selection, hopefully rare)
fraclogfile = 'p3-frac-3a.log'  # log pixels with fraction problem (nans within data); shouldn't be super rare

## qv is geotiff under analysis (I think 'qv' was me abbrev. 'query variable' originally)
qv = 'https://chs.coast.noaa.gov/htdata/raster2/elevation/Chesapeake_Coned_update_DEM_2016_8656/' \
     'Chesapeake_Coned_update_DEM_2016_EPSG-26918.vrt'


import logging
fmtstr = 'CB phase3: %(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=logfile, filemode='a+', format=fmtstr, datefmt='%y-%m-%d %H:%M')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
call = ' '.join(sys.argv)
logger.info("The script, called as '%s', started" % call)

#bnd log
bf1 = open(bndlogfile, 'a+')

#frac log
ff1 = open(fraclogfile, 'a+')

def crsproj(shpcrs, newcrs):
    import pyproj
    shp = pyproj.CRS(f'EPSG:{shpcrs}')
    # conedcrs = conedobj.rio.crs.to_epsg()
    new1 = pyproj.CRS(f'EPSG:{newcrs}')
    project = pyproj.Transformer.from_crs(shp, new1, always_xy=True).transform
    return project

def shpplt(shp, crs1=None, trans=None, **kwargs):
    import geopandas as gpd
    from cartopy import crs
    shpp = gpd.GeoSeries(shp)
    if crs1 == 'cpc':
        cpc = crs.PlateCarree()
        if trans is not None:
            t1 = crsproj(str(trans), '4326')
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
    dcomp = rio.open_rasterio('ATL_composite_veg_cmp.tif')
    # open stream to online CoNED gtiff
    dqv = rio.open_rasterio(qv)
    d8 = rio.open_rasterio('L8_2018_ATL_Unveg_Veg_Water_UVVR.tif')

# geotiff (gt) boundaries, to subset the dcomp and the Landsat geotiff to
gtlx = dqv.x[0].values
gtux = dqv.x[-1].values
gtly = dqv.y[-1].values
gtuy = dqv.y[0].values

gtp1 = cdg.poly_region((gtlx, gtly, gtux, gtuy), False)
# shpplt(gtp1,'cpc',dqv.rio.crs.to_epsg())
# gca().coastlines()


to_4326 = crsproj(dqv.rio.crs.to_epsg(), dcomp.rio.crs.to_epsg())
to_dqv = crsproj(dcomp.rio.crs.to_epsg(), dqv.rio.crs.to_epsg())

gtp2 = transform(to_4326, gtp1)

# shpplt(gtp2,'cpc')
# gca().coastlines()

lsp1 = cdg.poly_region(gtp2.bounds, False)
# lsp1p = gpd.GeoSeries(lsp1)
# ax1 = gca()
# lsp1p.plot(ax=ax1,fc='None', ec='red')


lslx = np.where(dcomp.x > gtp2.bounds[0])[0][0]
lsux = np.where(dcomp.x < gtp2.bounds[2])[0][-1]
lsly = np.where(dcomp.y > gtp2.bounds[1])[0][-1]
lsuy = np.where(dcomp.y < gtp2.bounds[3])[0][0]

dcomp1 = dcomp[0, lsuy:lsly, lslx:lsux]
dx,dy = dcomp1.rio.resolution()

logger.info(f'lsuy: {lsuy} lsly: {lsly} lslx: {lslx} lsux: {lsux}')

###########################################################
## create output file, of the same size
from osgeo import gdal

gt2 = (float(dcomp1.x[0] - dx / 2), dx, 0, float(dcomp1.y[0] - dy / 2), 0, dy)

driver = gdal.GetDriverByName("GTiff")
# outdata = driver.Create('nv/test.tif', int(dcomp1.x.size), int(dcomp1.y.size), 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata = driver.Create('CB_tst2.tif', dcomp1.x.size, dcomp1.y.size, 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'INTERLEAVE=PIXEL'])
outdata.SetGeoTransform(gt2)
outdata.SetProjection(dcomp1.spatial_ref.attrs['spatial_ref'])
outdata.SetMetadata({'AREA_OR_POINT': 'Area'})

# initialize, in a fashion, with nan
band = outdata.GetRasterBand(1)
band.SetNoDataValue(-np.inf)
band.SetMetadata(['RepresentationType=ATHEMATIC'])
band.Fill(np.nan)
outdata.FlushCache()

############################

nx = dcomp1.shape[1] // 100
ny = dcomp1.shape[0] // 100

# j = 21
# i = 97

# j = 22
# i = 96
icount = 0
for j in range(ny):
    for i in range(nx):
        sluy = j*100
        sllx = i*100
        if j == ny-1:
            slly = None
        else:
            slly = (j+1)*100
        if i == nx-1:
            slux = None
        else:
            slux = (i+1)*100
        sl_y = slice(sluy,slly)
        sl_x = slice(sllx,slux)
        curblk1 = dcomp1[sl_y,sl_x]
        cb1sz = curblk1.size
        idx1 = np.where((curblk1 > -10) & (curblk1 != 0))
        gdct1 = len(idx1[0])

        if gdct1 == 0:
            continue

        logger.info(f'j: {j} :: i: {i} :: size: {cb1sz} :: gdct: {gdct1}')

        curblk_gd = curblk1[xr.DataArray(idx1[0]), xr.DataArray(idx1[1])]
        cblx = curblk_gd.x.min().values
        cbly = curblk_gd.y.min().values
        cbux = curblk_gd.x.max().values
        cbuy = curblk_gd.y.max().values

        cbp1 = cdg.poly_region((cblx-dx, cbly+dy, cbux+dx, cbuy-dy), False)
        cbp2 = transform(to_dqv, cbp1)

        ## if the converted landsat box (containing valid pixels) lies outside of the whole CoNED dataset,
        ## continue (this can happen near boundaries)
        if (cbp2.bounds[0] > dqv.x[-1]) | (cbp2.bounds[1] > dqv.y[0]) | (cbp2.bounds[2] < dqv.x[0]) | \
            (cbp2.bounds[3] < dqv.y[-1]):
            continue

        qvlx = np.where(dqv.x > cbp2.bounds[0])[0][0]
        qvux = np.where(dqv.x < cbp2.bounds[2])[0][-1]
        qvly = np.where(dqv.y > cbp2.bounds[1])[0][-1]
        qvuy = np.where(dqv.y < cbp2.bounds[3])[0][0]

        dqv1 = dqv[0, qvuy:qvly, qvlx:qvux]

        valgrab = False  # value grab flag: means that I successfully got data through the stream
        vgct = 0  # a counter to verify I don't try to reconnect more than 5 times
        while valgrab is False:  # of course, the condition has to be true to enter into the loop, thus, False to start with
            try:   # if you aren't aware of try/except blocks, read up on em
                dqv1max = dqv1.max()  # this will (finally) request the actual values from the data server, b/c it's an operation on em!
                valgrab = True  # if the try part is successful as a whole (the above is the only part that can fail), set True
            except:
                if vgct > 4:  # if we got here for the 5th time, stop trying
                    break  # leave the while loop
                else:
                    logger.info('failed datagrab') # write out that it failed (want to track if this happens "too much")
                    time.sleep(15)  # wait a few seconds, i.e., don't just hammer the server with 5 requests too quickly
                    dqv = rio.open_rasterio(qv)  # try to (re-)establish a connection to the CoNED server (defined in 'qv' as a string URL)
                    dqv1 = dqv[0, qvuy:qvly, qvlx:qvux]  # re-slice the data (this apparently is important, the other 'slice' may be orphaned)
                    vgct = vgct + 1  # count this as one attempt; if it fails to get the values in the next iter of the try part, we'll attempt another reconnection up to the limit
                    continue  # continue with the next iteration of the while loop; i.e., do the try / except block again


        if dqv1max == dqv._FillValue:
            continue

        idq = np.where(dqv1 == dqv._FillValue)
        if not len(idq[0]) == 0:
            dqv1[xr.DataArray(idq[0]), xr.DataArray(idq[1])] = np.nan

        for ii in range(len(idx1[0])):
            xd = np.double(curblk_gd[ii].x)
            yd = np.double(curblk_gd[ii].y)
            vgfrac1 = float(curblk_gd[ii])

            xd_ll = float(xd - dx / 2)
            yd_ll = float(yd - dx / 2)
            xd_ur = float(xd + dx / 2)
            yd_ur = float(yd + dx / 2)

            p1 = cdg.poly_region((yd_ll, xd_ll, yd_ur, xd_ur))
            p2 = transform(to_dqv, p1)
            p2p = gpd.GeoSeries(p2)
            x1a, y1a, x2a, y2a = p2.bounds

            xstarta = np.floor(x1a) - 1
            ystarta = np.floor(y1a) - 1
            xexta = x2a - x1a + 1
            yexta = y2a - y1a + 1
            outshpa = (int(np.ceil(yexta) + 1), int(np.ceil(xexta) + 1))

            # here, check to see if we ran into a CoNED tif or selection bound
            if ((ystarta + 0.5) < dqv1.y[-1]) or ((ystarta + np.ceil(yexta) + 0.5) > dqv1.y[0]) \
                    or ((xstarta + 0.5) < dqv1.x[0]) or ((xstarta + np.ceil(xexta) + 0.5) > dqv1.x[-1]):
                bf1.write(f'j: {j} i: {i} idx10: {idx1[0][ii]} idx11: {idx1[1][ii]}\n')
                bf1.flush()
                continue

            trns1a = Affine(1, 0, xstarta, 0, 1, ystarta)
            trnsfmr1a = AffineTransformer(trns1a)

            p2rasa = rasterize([p2], out_shape=outshpa, transform=trns1a)
            p2ras1a = np.ma.masked_where(p2rasa == 0, p2rasa)

            x3a = trnsfmr1a.xy(0, np.arange(outshpa[1], dtype=np.int32))[0]
            y3a = trnsfmr1a.xy(np.arange(outshpa[0], dtype=np.int32), 0)[1]

            ixmin = int((xstarta + 0.5) - dqv1.x[0])
            ixmax = int((xstarta + 0.5) - dqv1.x[0] + np.ceil(xexta) + 1)
            iymin = int(dqv1.y[0] - ystarta + 0.5)
            iymax = int(dqv1.y[0] - ystarta + 0.5 - np.ceil(yexta) - 1)
            qvsl1 = dqv1[iymax:iymin, ixmin:ixmax]
            qvsl2 = qvsl1.values

            # multiply landsat region raster times same region CoNED raster to get elevations inside landsat pixel region
            # then strip out masked values (places where the landsat pixel region is not in the raster),
            # and strip out nans in the CoNED data (where they were fill values, see above ~line 211), this will
            # be tested below
            ls_coned_elv = p2ras1a * qvsl2[::-1]
            ls_coned_elv1 = ls_coned_elv[~ls_coned_elv.mask].data
            ls_coned_elv2 = ls_coned_elv1[~np.isnan(ls_coned_elv1)]
            ls_coned_elv2.sort()

            # if vegfrac1 == 1, then write out the mean of all coned elevs
            # within the pixel
            if vgfrac1 == 1:
                vfmnelev  = float(np.mean(ls_coned_elv2))
                band.WriteArray(np.array([[vfmnelev]]), int(i*100+idx1[1][ii]), int(j*100+idx1[0][ii]))
                icount = icount + 1
                if (icount % 500) == 0:
                    outdata.FlushCache()
                    print('wrote another 500\n')
                    sys.stdout.flush()
                continue
            
            # test how big the fraction is of # of coned elvs over # ls pix blocks in coned space (sq m)
            # starting with 0.95 as minimum cutoff at first

            frac_coverage = ls_coned_elv2.size / p2ras1a.sum()
            if frac_coverage < 0.95:
                ff1.write(f'j: {j} i: {i} idx10: {idx1[0][ii]} idx11: {idx1[1][ii]} frac: {frac_coverage}\n')
                ff1.flush()
                continue

            # veg fraction of number of points in CoNED for shape
            vf_conpts = ls_coned_elv2.size * vgfrac1

            # mean of highest 'vegetated fraction' of sorted CoNED elevations
            vfmnelev = float(np.mean(ls_coned_elv2[-int(vf_conpts):]))

            ## write to output file, remember, x first!  Only flush every thousand writes.
            band.WriteArray(np.array([[vfmnelev]]), int(i*100+idx1[1][ii]), int(j*100+idx1[0][ii]))
            icount = icount + 1
            if (icount % 500) == 0:
                outdata.FlushCache()
                print('wrote another 500\n')
                sys.stdout.flush()

outdata.FlushCache()
sys.stdout.flush()
logger.info('Finished\n')
logger.info('-'*50 + '\n\n')
