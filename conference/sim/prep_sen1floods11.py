"""Builds real inference traces from Sen1Floods11 (hand-labelled Sentinel-2 chips).
Each chip (512x512 px, 10 m) is scanned by a 64x64 footprint with stride 32 along a
boustrophedon path, one footprint per slot. A footprint is 'flooded' (X=1) if >=10% of
its labelled pixels are water. Two classifiers are trained on the train split and
calibrated (isotonic) on the validation split:
  on-board: logistic regression on 8x8 average-pooled RGB+NIR (257 parameters)
  ground  : gradient-boosted trees on per-band and water-index statistics, 13 bands
Output: traces.npz with, for the test chips, per slot: chip id, X, on-board prob, ground prob.
Usage: python3 -I prep_sen1floods11.py <data_root> <out.npz>"""
import sys, os, csv, numpy as np, tifffile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
root, out = sys.argv[1], sys.argv[2]
W, S, THR = 64, 32, 0.10

def chips(split):
    with open(os.path.join(root, 'flood_handlabeled', f'flood_{split}_data.csv')) as f:
        return [r[0].replace('_S1Hand.tif', '') for r in csv.reader(f) if r]

def path_windows():
    n = (512 - W)//S + 1
    for i in range(n):
        cols = range(n) if i % 2 == 0 else range(n-1, -1, -1)
        for j in cols: yield i*S, j*S

def feats(name):
    img = tifffile.imread(os.path.join(root, 'S2Hand', name + '_S2Hand.tif')).astype(np.float32)
    if img.shape[0] != 13: img = np.moveaxis(img, -1, 0)
    lab = tifffile.imread(os.path.join(root, 'LabelHand', name + '_LabelHand.tif'))
    img = np.nan_to_num(img) / 10000.0
    g, r, nir, sw = img[2], img[3], img[7], img[11]
    ndwi = (g - nir)/(g + nir + 1e-6); mndwi = (g - sw)/(g + sw + 1e-6)
    F1, F2, X = [], [], []
    for y0, x0 in path_windows():
        L = lab[y0:y0+W, x0:x0+W]; valid = L >= 0
        if valid.mean() < 0.8 or (img[:, y0:y0+W, x0:x0+W].sum(0) == 0).mean() > 0.2: continue
        X.append(int((L[valid] == 1).mean() >= THR))
        blkimg = np.clip(img[[1, 2, 3, 7], y0:y0+W, x0:x0+W], 0, None)
        F1.append(blkimg.reshape(4, 8, 8, 8, 8).mean((2, 4)).ravel())   # raw 8x8 means, bands B,G,R,NIR
        blk = [img[b, y0:y0+W, x0:x0+W] for b in range(13)] + [ndwi[y0:y0+W, x0:x0+W], mndwi[y0:y0+W, x0:x0+W]]
        F2.append(np.concatenate([[b.mean(), b.std(), *np.percentile(b, [10, 50, 90])] for b in blk] +
                                 [[(blk[13] > 0).mean(), (blk[14] > 0).mean()]]))
    return np.array(F1), np.array(F2), np.array(X)

def load(split):
    D = [feats(n) for n in chips(split)]
    ids = np.concatenate([np.full(len(d[2]), k) for k, d in enumerate(D)])
    return [np.concatenate([d[i] for d in D if len(d[2])]) for i in range(3)] + [ids]

CACHE = out + '.feats.pkl'
if os.path.exists(CACHE):
    import pickle; tr, va, te = pickle.load(open(CACHE, 'rb'))
else:
    import pickle; tr, va, te = load('train'), load('valid'), load('test'); pickle.dump((tr, va, te), open(CACHE, 'wb'))
print('windows train/valid/test', len(tr[2]), len(va[2]), len(te[2]), 'flood share', tr[2].mean().round(3), te[2].mean().round(3))
SD = tr[0].reshape(-1, 4, 64).std((0, 2))   # per-band std of the 8x8 means (training split)
def onboard_feats(F, bands, k, noise=0.0, seed=0):
    """k x k average pooling of the selected bands from the stored 8x8 means, log-scaled.
    noise > 0 adds Gaussian sensor noise with std noise*SD[band] to every 8x8 cell (fixed seed)."""
    A = F.reshape(-1, 4, 8, 8).copy()
    if noise > 0:
        A += np.random.default_rng(seed).normal(0, 1, A.shape)*noise*SD[None, :, None, None]
        A = np.clip(A, 0, None)
    A = A[:, bands]
    A = A.reshape(len(F), len(bands), k, 8//k, k, 8//k).mean((3, 5))
    return np.log1p(10*A.reshape(len(F), -1))
VARIANTS = {'on8': ([0, 1, 2, 3], 8, 0.0), 'n2': ([0, 1, 2, 3], 8, 2.0), 'n4': ([0, 1, 2, 3], 8, 4.0), 'n6': ([0, 1, 2, 3], 8, 6.0), 'n8': ([0, 1, 2, 3], 8, 8.0)}
iso = lambda p, y: IsotonicRegression(out_of_bounds='clip', y_min=1e-3, y_max=1-1e-3).fit(p, y)
res = {}
for k, (bands, kk, nz) in VARIANTS.items():
    ftr, fva, fte = (onboard_feats(d[0], bands, kk, nz, sd) for sd, d in enumerate((tr, va, te)))
    m = LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced').fit(ftr, tr[2])
    c = iso(m.predict_proba(fva)[:, 1], va[2])
    res[k+'_va'] = c.predict(m.predict_proba(fva)[:, 1]); res[k+'_te'] = c.predict(m.predict_proba(fte)[:, 1])
    print(k, 'params', ftr.shape[1]+1, 'test acc', ((res[k+'_te'] >= .5) == te[2]).mean().round(4), 'frac pos', (res[k+'_te'] >= .5).mean().round(3))
m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[1], tr[2])
c2 = iso(m2.predict_proba(va[1])[:, 1], va[2])
res['gr_va'] = c2.predict(m2.predict_proba(va[1])[:, 1]); res['gr_te'] = c2.predict(m2.predict_proba(te[1])[:, 1])
print('ground test acc', ((res['gr_te'] >= .5) == te[2]).mean().round(4))
Xs, ids = tr[2], tr[3]; same = ids[1:] == ids[:-1]
x0, x1 = Xs[:-1][same], Xs[1:][same]
p01 = ((x0 == 0) & (x1 == 1)).sum()/(x0 == 0).sum(); p10 = ((x0 == 1) & (x1 == 0)).sum()/(x0 == 1).sum()
print('p01 p10', p01.round(4), p10.round(4))
np.savez(out, chip=te[3], X=te[2], chip_va=va[3], X_va=va[2], p01=p01, p10=p10, **res)
