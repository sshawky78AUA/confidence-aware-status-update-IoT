"""Fig: (a) reliability of on-board models, (b) error rates vs on-board confidence, (c) sample path.
Usage: python fig_inference.py <traces.npz> <out.pdf>"""
import sys, numpy as np
from figstyle import plt, OI, PAGE_W
d = np.load(sys.argv[1])
X = d['X'].astype(int)
def ece(s, x, nb=10):
    b = np.minimum((s*nb).astype(int), nb-1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean()*abs(s[m].mean()-x[m].mean())
    return e
fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 1.95), gridspec_kw={'width_ratios': [1, 1, 1.6]})
edges = np.linspace(0, 1, 11)
for key, col, lab in [('on8', OI['verm'], 'on-board'), ('n8', OI['blue'], 'noisy $8\\sigma$'), ('gr', OI['green'], 'ground')]:
    s = d[key+'_te']; b = np.minimum((s*10).astype(int), 9)
    xs = [s[b == k].mean() for k in range(10) if (b == k).sum() > 30]
    ys = [X[b == k].mean() for k in range(10) if (b == k).sum() > 30]
    ax[0].plot(xs, ys, marker='o', color=col, label=f'{lab}, ECE {100*ece(s,X):.1f}%')
    print(key, 'ECE', ece(s, X), 'acc', ((s >= .5) == X).mean())
ax[0].plot([0, 1], [0, 1], ':', color=OI['grey'], lw=0.8)
ax[0].set_xlabel('predicted flood probability'); ax[0].set_ylabel('fraction truly flooded'); ax[0].legend(loc='lower right', fontsize=6, handlelength=1.0, handletextpad=0.4, labelspacing=0.25, borderpad=0.3, borderaxespad=0.2, frameon=True, framealpha=0.92, edgecolor='none')
ax[0].set_title('(a)', loc='left', fontweight='bold', pad=3); ax[0].set_title('reliability', loc='right', fontsize=7, pad=3)
s = d['on8_te']; c = np.maximum(s, 1-s); lab = (s >= .5).astype(int); g = (d['gr_te'] >= .5).astype(int)
cb = np.array([.5, .6, .7, .8, .9, .95, .99, 1.0001]); mids = []; eon = []; egr = []; share = []
for lo, hi in zip(cb[:-1], cb[1:]):
    m = (c >= lo) & (c < hi)
    mids.append((lo+min(hi, 1))/2); eon.append((lab[m] != X[m]).mean()); egr.append((g[m] != X[m]).mean()); share.append(m.mean())
ax[1].plot(mids, eon, 'o-', color=OI['verm'], label='on-board label')
ax[1].plot(mids, 1-np.array(mids), ':', color=OI['verm'], lw=0.8, label='$1-c$ (calibrated)')
ax[1].plot(mids, egr, 's-', color=OI['green'], label='ground label, $\\varepsilon_{\\rm g}(c)$')
ax[1].set_xlabel('on-board confidence $c$'); ax[1].set_ylabel('error probability'); ax[1].legend(fontsize=6.2, handlelength=1.4)
ax[1].set_title('(b)', loc='left', fontweight='bold', pad=3); ax[1].set_title('errors vs. confidence', loc='right', fontsize=7, pad=3); print('bins', list(zip(mids, eon, egr, share)))
# sample path: chip with a transition
chip = d['chip']; CH, ST = 65, 80   # chip/offset chosen for few state changes and a visible error burst
best = np.where(chip == CH)[0][ST:ST+120]
t = np.arange(len(best))
ax[2].fill_between(t, 0, X[best], step='mid', color=OI['sky'], alpha=0.35, lw=0, label='flooded')
ax[2].plot(t, s[best], color=OI['verm'], lw=0.8, label='flood probability')
wr = lab[best] != X[best]
ax[2].plot(t[wr], s[best][wr], 'x', color=OI['black'], ms=3, label='wrong label')
ax[2].axhline(0.5, color=OI['grey'], lw=0.6, ls=':')
ax[2].set_xlabel('slot (one footprint per slot)'); ax[2].set_ylabel('flood probability'); ax[2].set_ylim(-0.02, 1.42); ax[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[2].legend(ncol=3, loc='upper center', fontsize=6.2, handlelength=1.4, columnspacing=1.5, borderaxespad=0.4, frameon=True, framealpha=1, edgecolor='none'); ax[2].set_title('(c)', loc='left', fontweight='bold', pad=3); ax[2].set_title('on-board errors come in runs', loc='right', fontsize=7, pad=3)
print('chip', chip[best[0]], 'flooded', X[best].mean(), 'errors', wr.sum(), 'runs', np.diff(np.r_[0, wr.astype(int), 0]).clip(0).sum())
fig.tight_layout(w_pad=0.6); fig.savefig(sys.argv[2])
