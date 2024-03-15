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
from matplotlib import cm


def crsproj(shpcrs, newcrs):
    import pyproj
    shp = pyproj.CRS(f'EPSG:{shpcrs}')
    # conedcrs = conedobj.rio.crs.to_epsg()
    new1 = pyproj.CRS(f'EPSG:{newcrs}')
    project = pyproj.Transformer.from_crs(shp, new1, always_xy=True).transform
    return project


def shpplt(shp, crs1=None, **kwargs):
    import geopandas as gpd
    from cartopy import crs
    shpp = gpd.GeoSeries(shp)
    if crs1 == 'cpc':
        cpc = crs.PlateCarree()
        t1 = crsproj('3857', '4326')
        shp1a = transform(t1, shp)
        shp1ap = gpd.GeoSeries(shp1a)
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


if 'd5' not in locals():
    vf1 = gpd.read_file('CMU_CB_Veg/CMU_CB_Veg.shp')
    uvf1 = gpd.read_file('CMU_CB_unveg/CMU_CB_unveg.shp')

    slice1 = slice(0, None)
    vf1a = vf1[slice1]
    uvf1a = uvf1[slice1]

    fid_uv = uvf1a.FID_CMU.values
    idx1a = []
    for id in fid_uv:
        idx = np.where(vf1a.FID_CMU == id)[0]
        if len(idx) > 0:
            idx1a.append(idx[0])

    vf1b = vf1a.iloc[idx1a]
    # vf1b = vf1b.reindex(index=arange(7822,dtype=int64))
    fid_v = np.int64(vf1b.FID_CMU.values)
    idx1b = []
    for id in fid_v:
        idx = np.where(uvf1a.FID_CMU == id)[0]
        if len(idx) > 0:
            idx1b.append(idx[0])

    uvf1b = uvf1a.iloc[idx1b]

    huh = (pd.concat([vf1b, uvf1b])).dissolve('FID_CMU')

    # open dbf file for elevation data per mu shape
    df1 = gpd.read_file('mu_elev_CB.dbf')

    d5 = rio.open_rasterio('L8_2018_ATL_Unveg_Veg_Water_UVVR.tif')

cmuproj = '3857'
# open stream to online CoNED gtiff
dqv = rio.open_rasterio(
    'https://chs.coast.noaa.gov/htdata/raster2/elevation/Chesapeake_Coned_update_DEM_2016_8656/Chesapeake_Coned_update_DEM_2016_EPSG-26918.vrt')
prj1 = crsproj(cmuproj, dqv.rio.crs.to_epsg())  # CMU crs to CoNED crs; might not need in this script
prj2 = crsproj(dqv.rio.crs.to_epsg(), '4326')  # probably don't need this CoNED UTM to LandSat crs transform
prj3 = crsproj('4326', dqv.rio.crs.to_epsg())  # transform to give the UTM coords of the LandSat data pixels
prj4 = crsproj(cmuproj, '4326')  # convert CMU shape to LandSat crs to get relevant, overlying LandSat pixels

noplot = True
notest = True
biggradtest = 1.

# fid1 = 3

npoly = huh.shape[0]
dx, dy = d5.rio.resolution()
pdout = pd.DataFrame()

st = time.time()

for i in range(0, npoly):
    # i = np.where(huh.index == fid1)[0][0]
    st1 = time.time()
    cur1 = huh.iloc[i]  # current db entry
    shp1 = cur1.geometry
    c1flg = cur1.FLG

    fid = cur1.name

    # veg fraction from shape areas
    vgfrac = cur1.APGN_M2 / cur1.ATOT_M2

    ## change back to veg fraction when running on full shape!
    vgarea = cur1.APGN_M2

    shp1p = gpd.GeoSeries(shp1)
    # muelev = df1.MU_ELEV[df1.FID_CMU == fid].values[0]
    vgelev = df1.VG_ELEV[df1.FID_CMU == fid].values[0]

    # this check will possibly exclude some locations
    if vgelev <= -999.0:
        continue

    ## shpx is the transformed CMU poly in the Lat/Lon (4326) crs
    ## used to match the relevant LandSat pixels overlying the CMU polygon
    shpx = transform(prj4, shp1)
    shpxp = gpd.GeoSeries(shpx)
    x1, y1, x2, y2 = shpx.bounds

    iflx = np.where(d5.x < x1)[0][-1]
    ifly = np.where(d5.y < y1)[0][0]

    floorx = d5.x[iflx]
    floory = d5.y[ifly]

    ceilx = d5.x[np.where(d5.x > x2)[0][0]]
    ceily = d5.y[np.where(d5.y > y2)[0][-1]]

    xstart = float(floorx - dx * 1.5)
    ystart = float(floory - dx * 1.5)
    xext = int(np.round((ceilx - floorx) / dx))
    yext = int(np.round((ceily - floory) / dx))
    outshp = (yext + 2, xext + 2)

    yidx = np.arange(ifly + 1, ifly + 1 - outshp[0], -1)  # for making a list of indices matching the raster
    xidx = np.arange(iflx - 1, iflx - 1 + outshp[1])  # for making a list of indices matching the raster

    trns1 = Affine(dx, 0, xstart, 0, dx, ystart)
    trnsfmr1 = AffineTransformer(trns1)

    shpxras = rasterize([shpx], out_shape=outshp, transform=trns1)
    shpxras1 = np.ma.masked_where(shpxras == 0, shpxras)

    x3 = trnsfmr1.xy(0, np.arange(outshp[1], dtype=np.int32))[0]
    y3 = trnsfmr1.xy(np.arange(outshp[0], dtype=np.int32), 0)[1]

    if not noplot:
        xg = np.arange(xstart, xstart + (xext + 2) * dx, dx)
        yg = np.arange(ystart, ystart + (yext + 1) * dx, dx)

        plt.figure()
        plt.pcolormesh(x3, y3, shpxras1)
        ax1 = plt.gca()
        shpxp.plot(ax=ax1, color='green', alpha=0.5)

        ax1.set_yticks(yg)
        ax1.set_xticks(xg)
        ax1.grid()
        plt.title(f'fid: {fid}')

    iy1, ix1 = np.where(shpxras1)  # indices where the landsat pixels of interest are

    if len(iy1) == 0:
        continue
    ## Okay, so the iy1 and ix1 are the indices of the yidx and xidx indices of the rasterized shape shp1
    ## in the lat/lon space of the Landsat pixels, where the Landsat pixels lie within shp1 (to a sufficient degree;
    ## i.e., the midpoint).  That is iy1 and ix1's purpose, as seen below.

    ## Now to select the raster's area that includes only a bit more than the shape,
    ## build a polygon with the limits of the area, and convert to CoNED's crs.
    ## This uses the iy1, ix1 values to select out the region to define.

    ## p1 is the lat-lon side of the polygon within which to grab
    ## all CoNED points that are around / within the current CMU shape
    p1_min_x = float(d5.x[xidx[ix1]].min() - dx)
    p1_max_x = float(d5.x[xidx[ix1]].max() + dx)
    p1_min_y = float(d5.y[yidx[iy1]].min() - dx)
    p1_max_y = float(d5.y[yidx[iy1]].max() + dx)

    p1 = cdg.poly_region((p1_min_y, p1_min_x, p1_max_y, p1_max_x))
    p1p = gpd.GeoSeries(p1)

    if not noplot:
        ax1 = plt.gca()
        p1p.plot(ax=ax1, alpha=0.3)

    p2 = transform(prj3, p1)
    p2p = gpd.GeoSeries(p2)

    x1a, y1a, x2a, y2a = p2.bounds
    # here, check to see if we ran into a tif bound or if the poly is outside the tif completely
    if (y1a < dqv.y[-1]) or (y2a > dqv.y[0]) or (x1a < dqv.x[0]) or (x2a > dqv.x[-1]):
        print('region on geotiff boundary\n')
        continue

    xstarta = np.floor(x1a) - 1
    ystarta = np.floor(y1a) - 1
    xexta = x2a - x1a
    yexta = y2a - y1a
    outshpa = (int(np.ceil(yexta) + 1), int(np.ceil(xexta) + 1))

    trns1a = Affine(1, 0, xstarta, 0, 1, ystarta)
    trnsfmr1a = AffineTransformer(trns1a)

    shp1a = transform(prj1, shp1)
    shp1rasa = rasterize([shp1a], out_shape=outshpa, transform=trns1a)
    shp1ras1a = np.ma.masked_where(shp1rasa == 0, shp1rasa)

    x3a = trnsfmr1a.xy(0, np.arange(outshpa[1], dtype=np.int32))[0]
    y3a = trnsfmr1a.xy(np.arange(outshpa[0], dtype=np.int32), 0)[1]

    ixmin = int((xstarta + 0.5) - dqv.x[0])
    ixmax = int((xstarta + 0.5) - dqv.x[0] + np.ceil(xexta) + 1)
    iymin = int(dqv.y[0] - ystarta + 0.5)
    iymax = int(dqv.y[0] - ystarta + 0.5 - np.ceil(yexta) - 1)
    qvsl1 = dqv[0, iymax:iymin, ixmin:ixmax]
    qvsl2 = qvsl1.values

    # track any large 'gradients' within the vegetated shape or LS pixels
    biggrad = None
    outq = shp1ras1a * qvsl2[::-1]
    outqmin = np.max([outq.min(),0])
    gradsz = outq.max() - outqmin
    bgshp = gradsz

    if not notest:
        print('\n')
        dx3max = np.float64(qvsl1.x.max() - x3a[-1])
        dx3min = np.float64(qvsl1.x.min() - x3a[0])
        dy3max = np.float64(qvsl1.y.max() - y3a[-1])
        dy3min = np.float64(qvsl1.y.min() - y3a[0])
        print(f'coned-min-x: {qvsl1.x.min().values}, rast-min-x: {x3a[0]}, delta: {dx3min}')
        print(f'coned-max-x: {qvsl1.x.max().values}, rast-min-x: {x3a[-1]}, delta: {dx3max}')
        print(f'coned-min-y: {qvsl1.y.min().values}, rast-min-y: {y3a[0]}, delta: {dy3min}')
        print(f'coned-max-y: {qvsl1.y.max().values}, rast-min-y: {y3a[-1]}, delta: {dy3max}')

    vgelevs = []
    vfnan = False
    lspxareas = []
    bgpx = None
    # ii=2
    for ii in range(iy1.size):
        xd = d5.x[xidx[ix1[ii]]]
        yd = d5.y[yidx[iy1[ii]]]
        vgfrac1 = float(d5[1, yidx[iy1[ii]], xidx[ix1[ii]]])
        if np.isnan(vgfrac1):
            vfnan = True
            continue
        elif vgfrac1 == 0.:
            continue

        xd_ll = float(xd - dx / 2)
        yd_ll = float(yd - dx / 2)
        xd_ur = float(xd + dx / 2)
        yd_ur = float(yd + dx / 2)

        p3 = cdg.poly_region((yd_ll, xd_ll, yd_ur, xd_ur))

        if not noplot:
            p3p = gpd.GeoSeries(p3)
            p3p.plot(ax=ax1, alpha=0.6, color='pink')

        p4 = transform(prj3, p3)

        if not noplot:
            plt.figure()
            plt.pcolormesh(x3a, y3a, shp1ras1a, alpha=0.6)
            ax1 = plt.gca()
            p4p = gpd.GeoSeries(p4)
            p4p.plot(ax=ax1, alpha=0.6)
            plt.title(f'fid: {fid}, in CoNED crs (UTM, Z18N)')

        p4ras = rasterize([p4], out_shape=outshpa, transform=trns1a)
        p4ras1 = np.ma.masked_where(p4ras == 0, p4ras)

        if not noplot:
            plt.pcolormesh(x3a, y3a, p4ras1, cmap='spring_r', alpha=0.4)

        out1 = p4ras1 * qvsl2[::-1]
        out1a = out1.flatten()[~out1.flatten().mask]
        out1a.sort()

        # track any large 'gradients' within the vegetated shape or LS pixels
        gradpxsz = out1a.max() - out1a.min()
        if bgpx is not None:
            if gradpxsz > bgpx:
                bgpx = gradpxsz
        else:
            bgpx = gradpxsz


        # veg fraction of number of points in CoNED for shape
        vf_conpts = out1a.shape[0] * vgfrac1

        # mean of highest 'vegetated fraction' of sorted CoNED elevations
        vfmnelev = np.mean(out1a[-int(vf_conpts):])
        vgelevs.append(vfmnelev)

        # record area of landsat pixel
        lspxareas.append(p4ras1.count())

    if len(vgelevs) == 0:
        continue
    ls_vgelev = np.mean(vgelevs)
    # error in vegetated plain elevation estimate calculated here minus published value
    vfelv_err = ls_vgelev - vgelev

    lspxarea = np.sum(lspxareas)
    ##print(f'lspx area: {lspxarea} ;; vgarea = {vgarea}')

    if biggrad:
        c1flg = c1flg - 100000

    if ((lspxarea / cur1.ATOT_M2) < 0.6):
        c1flg = c1flg - 1000000

    # calculate Landsat pixel area coverage divided by total area
    lspcov = lspxarea / cur1.ATOT_M2

    # get the end time
    et1 = time.time()
    # get the execution time
    elapsed_time1 = et1 - st1

    ls_vgelev = np.mean(vgelevs)
    # error in vegetated plain elevation estimate calculated here minus published value
    vfelv_err = ls_vgelev - vgelev

    if pdout.shape[0] == 0:
        pdout['FID_CMU'] = [fid]  # CMU database identifier
        pdout['VG_ELEV'] = vgelev  # reported CMU vegetated elevation
        pdout['VG_FRAC'] = vgfrac  # reported CMU vegetated fraction
        pdout['VFNAN'] = vfnan  # one or more Landsat pixels contains a 'nan' vegetated fraction
        pdout['ATOT_M2'] = cur1.ATOT_M2  # reported total CMU area
        pdout['LS_VGELV'] = ls_vgelev  # Landsat estimated vegetated elevation
        pdout['ELV_ERR'] = vfelv_err  # Landsat estimated vgelev minus reported vgelev
        pdout['FLG'] = c1flg  # CMU database flag value
        pdout['LS_COVER'] = lspcov  # Landsat pixel area coverage as fraction of total CMU shape area
        pdout['SHP_GRAD'] = bgshp  # 'gradient' (really max minus min) within shape sent to CoNED CRS
        pdout['PX_GRAD'] = bgpx  # max 'gradient' within Landsat pixels for the CMU
        pdout['ELPSD_T'] = elapsed_time1  # elapsed time to calculate Landsat estimated vgelev
        pdout = pdout.set_index('FID_CMU')
    else:
        # 'FID_CMU', 'MU_ELEV', 'VG_ELEV', 'VG_FRAC', 'ATOT_M2', 'ELV_ERR'
        pdout.loc[fid] = {'VG_ELEV': vgelev, 'VG_FRAC': vgfrac, 'VFNAN': vfnan, 'ATOT_M2': cur1.ATOT_M2,
                          'LS_VGELV': ls_vgelev, 'ELV_ERR': vfelv_err, 'FLG': c1flg, 'LS_COVER': lspcov,
                          'SHP_GRAD': bgshp, 'PX_GRAD': bgpx, 'ELPSD_T': elapsed_time1}

    print(f'fid: {fid}; vfelv_err: {vfelv_err: .2f}; elapsed time: {elapsed_time1: .1f}\n')
    sys.stdout.flush()

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print(f'final elapsed time: {elapsed_time: .1f}\n')
sys.stdout.flush()
