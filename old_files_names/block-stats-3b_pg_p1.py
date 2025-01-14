p1 = pload('ng1_mn_all_but1_flags.pickle')
p2 = pload('p2n2-all.pickle')

# cur1 = p1.loc[p1.ELV_ERR>-10]
# figure(); plot(cur1.VG_FRAC, cur1.ELV_ERR, '.')
# cur1 = p1a.loc[p1a.ELV_ERR>-10]
# plot(cur1.VG_FRAC, cur1.ELV_ERR, '.',alpha=0.4)
#

p1a = p1.loc[p1.ELV_ERR > -10]
idx1a = np.argsort(p1a.VG_FRAC)

cur1 = p1a
idxa = idx1a
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='phase1', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p1a_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p1a_ef[0], p1a_ef[1], lw=1, label='phase1')
legend(loc=3)

savefig('phase2_plots/mae-phase1.png')


# -------

p2a = p2.loc[p2.ELV_ERR > -10]
idx2a = np.argsort(p2a.VG_FRAC)

cur1 = p2a
idxa = idx2a
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='phase2', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2a_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2a_ef[0], p2a_ef[1], lw=1, label='phase2 fit')
legend(loc=3)

savefig('phase2_plots/mae-phase2.png')

# -------

p2b = p2a.loc[~p2a.VFNAN]
idx2b = np.argsort(p2b.VG_FRAC)

cur1 = p2b
idxa = idx2b
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, nonan', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2b_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2b_ef[0], p2b_ef[1], lw=1, label='p2, nonan fit')
legend(loc=3)

savefig('phase2_plots/mae-nonan.png')

# -------

p2c = p2a.loc[(p2a.ELV_ERR > -1) & (p2a.ELV_ERR < 1)]
idx2c = np.argsort(p2c.VG_FRAC)

cur1 = p2c
idxa = idx2c
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, abs(err) < 1', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2c_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2c_ef[0], p2c_ef[1], lw=1, label='p2, abs(err) < 1 fit')
legend(loc=3)

savefig('phase2_plots/mae-err1m.png')

# -------

p2d = p2a.loc[p2a.PX_GRAD < 13]
idx2d = np.argsort(p2d.PX_GRAD)

cur1 = p2d
idxa = idx2d
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.PX_GRAD.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel("max pixel 'gradient'")
ylabel('elevation estimate error (m)')
title('elevation estimate error by max pixel gradient')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2d_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2d_ef[0], p2d_ef[1], lw=1, label='p2, fit')
legend(loc=2)

savefig('phase2_plots/mae-pg.png')


# -------

p2e = p2a.loc[~p2a.VFNAN & (p2a.ELV_ERR > -1) & (p2a.ELV_ERR < 1)]
idx2e = np.argsort(p2e.PX_GRAD)

cur1 = p2e
idxa = idx2e
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.PX_GRAD.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, nonan, abs(err)<1', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel("max pixel 'gradient' (m)")
ylabel('elevation estimate error (m)')
title('elevation estimate error by max pixel gradient')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2d_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2d_ef[0], p2d_ef[1], lw=1, label='p2, nonan, abs(err)<1 fit')
legend(loc=2)

savefig('phase2_plots/mae-pg_nonan-err1m.png')


# -------

p2f = p2a.loc[~p2a.VFNAN & (p2a.ELV_ERR > -1) & (p2a.ELV_ERR < 1)]
idx2f = np.argsort(p2f.VG_FRAC)

cur1 = p2f
idxa = idx2f
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, nonan, abs(err)<1', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2f_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2f_ef[0], p2f_ef[1], lw=1, label='p2, nonan, abs(err)<1 fit')
legend(loc=3)

savefig('phase2_plots/mae_nonan-err1m.png')

# -------

p2g = p2a.loc[~p2a.VFNAN & (p2a.ELV_ERR > -1) & (p2a.ELV_ERR < 1) & (p2a.LS_COVER > 0.7)]
idx2g = np.argsort(p2g.VG_FRAC)

cur1 = p2g
idxa = idx2g
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, nonan, abs(err)<1, lsc>0.7', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2g_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2g_ef[0], p2g_ef[1], lw=1, label='p2, nonan, abs(err)<1, lsc>0.7 fit')
legend(loc=3)

savefig('phase2_plots/mae_lsc07m-nonan-err1m.png')

# -------

p2h = p2a.loc[~p2a.VFNAN & (p2a.ELV_ERR > -0.5) & (p2a.ELV_ERR < 0.5)]
idx2h = np.argsort(p2h.VG_FRAC)

cur1 = p2h
idxa = idx2h
eesz3 = cur1.ELV_ERR.size // 100
bind2b = np.zeros((eesz3 + 1, 2), dtype=np.float32)
for i in arange(0, (eesz3 + 1) * 100, 100):
    print(f'i: {i}; i+100: {i + 100}')
    bind2b[i // 100, 0] = cur1.VG_FRAC.iloc[idxa][i:i + 100].values.mean()
    bind2b[i // 100, 1] = (np.abs(cur1.ELV_ERR.iloc[idxa][i:i + 100].values)).mean()

figure(); plot(bind2b[:, 0], bind2b[:, 1], '+-', c='goldenrod', label='p2, nonan, abs(err)<.5m', lw=1)
axhline(0,c='black')
axhline(0.10,c='blue')

xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')

pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=2)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg2 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
pf1 = polyfit(bind2b[:, 0], bind2b[:, 1], deg=3)
pv1 = polyval(pf1, bind2b[:, 0])
pfdeg3 = np.sqrt(((bind2b[:, 1] - pv1) ** 2).sum() / pv1.size)
print(f'deg2: {pfdeg2:.6f} ;; deg3: {pfdeg3:.6f}')

## error fit (ef) function
p2h_ef = (bind2b[:, 0], polyval(pf1, bind2b[:, 0]))

plot(p2h_ef[0], p2h_ef[1], lw=1, label='p2, nonan, abs(err)<.5m fit')
legend(loc=3)

savefig('phase2_plots/mae_nonan-err05m.png')



#
#
#
# xlabel('pixel gradient (m)')
# ylabel('elevation estimate error (m)')
# title('elevation estimate error by pixel gradient')


plot(p1a_ef[0], p1a_ef[1], lw=1, label='phase1')
plot(p2a_ef[0], p2a_ef[1], lw=1, label='phase2 fit')
plot(p2b_ef[0], p2b_ef[1], lw=1, label='p2, nonan fit')
plot(p2c_ef[0], p2c_ef[1], lw=1, label='p2, abs(err) < 1 fit')
plot(p2f_ef[0], p2f_ef[1], lw=1, label='p2, nonan, abs(err)<1 fit')
plot(p2g_ef[0], p2g_ef[1], lw=1, label='p2, nonan, abs(err)<1, lsc>0.7 fit')
plot(p2h_ef[0], p2h_ef[1], lw=1, label='p2, nonan, abs(err)<.5m fit')

axhline(0,c='black')
axhline(0.10,c='blue')
xlabel('vegetated fraction, reported')
ylabel('elevation estimate error (m)')
title('elevation estimate error by veg fraction')
legend(loc=3)

