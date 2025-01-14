import cartopy.feature as cf
import rioxarray as rio
from shapely import geometry
from cartopy.io import shapereader as shpreader
from osgeo import osr
import geopandas as gpd
import coawst_dataget as cdg
from shapely.ops import transform
from rasterio.transform import AffineTransformer, Affine
from rasterio.features import rasterize
import pickle

def crsproj(shpcrs,conedobj):
    import pyproj
    shp = pyproj.CRS(f'EPSG:{shpcrs}')
    conedcrs = conedobj.rio.crs.to_epsg()
    utm = pyproj.CRS(f'EPSG:{conedcrs}')
    project = pyproj.Transformer.from_crs(shp, utm, always_xy=True).transform
    return project

vf1 = gpd.read_file('CMU_CB_Veg/CMU_CB_Veg.shp')
uvf1 = gpd.read_file('CMU_CB_unveg/CMU_CB_unveg.shp')

slice1 = slice(0,None)
vf1a = vf1[slice1]
uvf1a = uvf1[slice1]

fid_uv = uvf1a.FID_CMU.values
idx1a = []
for id in fid_uv:
    idx = np.where(vf1a.FID_CMU == id)[0]
    if len(idx) > 0:
        idx1a.append(idx[0])

vf1b = vf1a.iloc[idx1a]
#vf1b = vf1b.reindex(index=arange(7822,dtype=int64))
fid_v = np.int64(vf1b.FID_CMU.values)
idx1b = []
for id in fid_v:
    idx = np.where(uvf1a.FID_CMU == id)[0]
    if len(idx) > 0:
        idx1b.append(idx[0])

uvf1b = uvf1a.iloc[idx1b]

huh = (pd.concat([vf1b,uvf1b])).dissolve('FID_CMU')
blah = huh.iloc[0].geometry
blahp = gpd.GeoSeries(blah)
blahp.plot()

npoly = huh.shape[0]

proj = '3857'

# open dbf file for elevation data per mu shape
df1 = gpd.read_file('mu_elev_CB.dbf')

# open stream to online CoNED gtiff
dqv = rio.open_rasterio(
    'https://chs.coast.noaa.gov/htdata/raster2/elevation/Chesapeake_Coned_update_DEM_2016_8656/Chesapeake_Coned_update_DEM_2016_EPSG-26918.vrt')

prj1 = crsproj(proj, dqv)

pdout = gpd.GeoDataFrame()

flog = open('logfile.txt','w+')
datenow = time.ctime()
flog.write(datenow+'\n')
flog.flush()

# loop through all regions; that is, polygons found in the 'huh' dataset
# constructed above
arr_i = []
for i in range(npoly):
    shp1 = huh.iloc[i].geometry
    cur1 = huh.iloc[i]
    c1flg = cur1.FLG
    # this is for not using any locations with FLG containing -1
    # but this does nothing here since the 'huh' dataframe only contains
    # regions with non-zero vegetated fraction
    if str(c1flg)[-1] == '1':
        continue
    arr_i.append(i)
    fid = cur1.name

    # veg fraction from shape areas
    vgfrac = cur1.APGN_M2 / cur1.ATOT_M2

    shp1p = gpd.GeoSeries(shp1)
    muelev = df1.MU_ELEV[df1.FID_CMU == fid].values[0]
    vgelev = df1.VG_ELEV[df1.FID_CMU == fid].values[0]

    # this check will possibly exclude some locations
    if vgelev <= -999.0:
        continue

    shp1a = transform(prj1, shp1)
    x1, y1, x2, y2 = shp1a.bounds
    xstart = floor(x1) - 1
    ystart = floor(y1) - 1
    xext = x2 - x1
    yext = y2 - y1
    outshp = (int(ceil(yext) + 1), int(ceil(xext) + 1))

    trns1 = Affine(1, 0, xstart, 0, 1, ystart)
    trnsfmr1 = AffineTransformer(trns1)

    shp1ras = rasterize([shp1a], out_shape=outshp, transform=trns1)
    shp1ras1 = np.ma.masked_where(shp1ras == 0, shp1ras)

    x3 = trnsfmr1.xy(0, arange(outshp[1], dtype=int32))[0]
    y3 = trnsfmr1.xy(arange(outshp[0], dtype=int32), 0)[1]

    # here, check to see if we ran into a tif bound or if the poly is outside the tif completely
    if (y1 < dqv.y[-1]) or (y2 > dqv.y[0]) or (x1 < dqv.x[0]) or (x2 > dqv.x[-1]):
        continue

    ixmin = int(x3[0] - dqv.x[0])
    ixmax = int(x3[-1] - dqv.x[0] + 1)
    iymin = int(dqv.y[0] - y3[0] + 1)
    iymax = int(dqv.y[0] - y3[-1])
    qvsl1 = dqv[0, iymax:iymin, ixmin:ixmax]

    # select out the shape's raster mask of the CoNED data
    # ::-1 is because of the way qvsl1 is returned combined with out shp1ras1 was built
    out1 = shp1ras1 * qvsl1.values[::-1]
    out1a = out1.compress(out1.flatten())
    out1a.sort()


    # veg fraction of number of points in CoNED for shape
    vf_conpts = out1a.shape[0] * vgfrac

    # mean of highest 'vegetated fraction' of sorted CoNED elevations
    vfmedelev = np.mean(out1a[-int(vf_conpts):])

    # error in vegetated plain elevation estimate calculated here minus published value
    vfelv_err = vfmedelev - vgelev

    ilo = int(out1a.shape[0]*0.25)
    ihi = int(out1a.shape[0]*0.75)
    del50 = out1a[ihi] - out1a[ilo]



    if newg1.shape[0] == 0:
        newg1['FID_CMU'] = [fid]
        newg1['MU_ELEV'] = muelev
        newg1['VG_ELEV'] = vgelev
        newg1['VG_FRAC'] = vgfrac
        newg1['ATOT_M2'] = cur1.ATOT_M2
        newg1['ELV_ERR'] = vfelv_err
        newg1['FLG'] = c1flg
        newg1['DEL50'] = del50
        newg1 = newg1.set_index('FID_CMU')
    else:
        # 'FID_CMU', 'MU_ELEV', 'VG_ELEV', 'VG_FRAC', 'ATOT_M2', 'ELV_ERR'
        newg1.loc[fid] = {'MU_ELEV': muelev, 'VG_ELEV': vgelev, 'VG_FRAC': vgfrac, 'ATOT_M2': cur1.ATOT_M2,
                 'ELV_ERR': vfelv_err, 'FLG': c1flg, 'DEL50': del50}

    flog.write(f'fid: {fid}\n')
    flog.flush()

ng1b = pd.DataFrame(newg1)
ng1b.to_pickle('ng1_mn_all_but1_flags.pickle')
