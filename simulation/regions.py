"""Fig. 2: decisions of rule (13) vs. slots since last delivery and confidence, real parameters.
Usage: python3 regions.py [NU=0.25] [FIG_DIR]"""
import numpy as np, sim as sim2
from figstyle import plt, OI, COL_W
from matplotlib.colors import ListedColormap
import matplotlib.patches as mp
MP = sim2.model_params(sim2.P); LT = sim2.link_tables(sim2.P)
k = int(np.argmin(np.abs(LT['dist']-1200))); q1, q2, E1, E2 = LT['q1'][k], LT['q2'][k], LT['E1'][k], LT['E2'][k]
import sys; NU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25; PI = MP['pi']
p0, p1 = MP['p']; xh = 0; r, w = p0, p1; Dm = 300; lam_over_V = 2.0
def prior(A):   # belief A slots after a delivery of a label with confidence 0.95, no observations since
    b = np.zeros(Dm+1); b[0] = 0.95; b[1] = 0.05
    for _ in range(A):
        nb = np.zeros_like(b); nb[0] += b[0]*(1-r); nb[1] += b[0]*r
        nb[0] += b[1:].sum()*w; nb[2:] += b[1:-1]*(1-w); nb[-1] += b[-1]*(1-w); b = nb
    return b
def epsg(c): return MP['eps_e'][np.clip(np.digitize(c, MP['edges'])-1, 0, 6)]
As = np.arange(0, 31); cs = np.linspace(0.5, 0.999, 160)
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 1.55), sharey=True)
for j, d in enumerate([0, 1]):
    R = np.zeros((len(cs), len(As)))
    for jj, A in enumerate(As):
        b0 = prior(A)
        for i, c in enumerate(cs):
            Lp = c if d else 1-c; b = b0.copy(); b[0] *= ((1-Lp)/PI[xh])**NU; b[1:] *= (Lp/PI[1-xh])**NU; b /= b.sum()
            mu = 1-b[0]; S = np.dot(np.arange(1, Dm+1)+1, b[1:]); e = epsg(c)
            G1 = (S*(1-2*w)-(1-mu)*(1-2*r))*d; G2 = (1-e)*S*(1-2*w)-e*(1-mu)*(1-2*r)
            u = [0, q1*G1-lam_over_V*E1 if d else -1e9, q2*G2-lam_over_V*E2]
            R[i, jj] = int(np.argmax(u))
    ax[j].imshow(R, origin='lower', aspect='auto', cmap=ListedColormap(['#ececec', OI['blue'], OI['orange']]), vmin=0, vmax=2,
                 extent=[As[0], As[-1], cs[0], cs[-1]], interpolation='nearest')
    ax[j].set_title('(a)' if d == 0 else '(b)', loc='left', fontsize=8, fontweight='bold', pad=3); ax[j].set_title('label agrees with GS' if d == 0 else 'label disagrees with GS', loc='right', fontsize=7, pad=3); ax[j].grid(False)
    ax[j].set_xlabel('slots since last delivery')
ax[0].set_ylabel('on-board confidence $c$')
fig.legend(handles=[mp.Patch(facecolor='#ececec', edgecolor='#888888', linewidth=0.5, label='silent'), mp.Patch(color=OI['blue'], label='label'), mp.Patch(color=OI['orange'], label='raw upload')],
           loc='upper center', ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.78, wspace=0.08)
fig.savefig((sys.argv[2] if len(sys.argv) > 2 else 'figures') + '/regions.pdf')
print('q1 q2 E1 E2', q1, q2, E1, E2)
