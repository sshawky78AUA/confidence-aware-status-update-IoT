"""Sample path of CA-DPP at 0.5 mJ/slot: state, GS estimate, actions, AoII and belief mu_t.
Usage: python fig_path.py <out.pdf>"""
import sys, numpy as np
from figstyle import plt, OI, COL_W
import sim as S
chips = np.unique(S.TR['chip_all'])
r = S.run('cadpp', 0.5e-3, (0.01, 0.25, 0), seed=1, split=('cv', chips, chips, 1), log=True); L = r['log']
st = int(sys.argv[2]) if len(sys.argv) > 2 else 4990; w = slice(st, st+120); t = np.arange(120)
fig, ax = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
ax[0].fill_between(t, 0, L['X'][w], step='mid', color=OI['sky'], alpha=0.35, lw=0, label='flooded (true state)')
ax[0].step(t, L['Xh'][w], where='mid', color=OI['black'], lw=0.9, label='GS status')
a = L['a'][w]; ok = L['ok'][w]
for act, m, col, lab in ((1, 'v', OI['blue'], 'label sent'), (2, 'D', OI['orange'], 'raw upload')):
    sel = a == act
    ax[0].plot(t[sel], np.full(sel.sum(), 1.12), m, color=col, ms=3.2, mfc=col, label=lab)
ax[0].set_ylim(-0.05, 1.25); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(['dry', 'flooded']); 
ax[0].set_title('(a)', loc='left', fontweight='bold', pad=3, fontsize=8); ax[0].set_ylim(-0.08, 1.22)
ax[1].step(t, L['D'][w], where='mid', color=OI['blue'], lw=0.9, label='AoII $\\Delta(t)$ (left)')
ax[1].set_ylabel('AoII (slots)'); ax2 = ax[1].twinx(); ax2.grid(False)
ax2.plot(t, L['mu'][w], color=OI['verm'], lw=0.9, ls='--', label='belief $\\mu_t$ (right)'); ax2.set_ylim(0, 1); ax2.set_ylabel('$\\mu_t$ = P(GS wrong)')
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1+h2, l1+l2, fontsize=6.2, loc='upper left', bbox_to_anchor=(0.06, 1.0))
ax[1].set_xlabel('slot'); ax[1].set_title('(b)', loc='left', fontweight='bold', pad=3, fontsize=8)
print('labels', (a == 1).sum(), 'raw', (a == 2).sum(), 'delivered', ok.sum(), 'dist', L['dist'][w].min(), L['dist'][w].max(), 'blocked', L['blocked'][w].sum())
h0, l0 = ax[0].get_legend_handles_labels(); fig.legend(h0, l0, ncol=4, fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=0.8)
fig.tight_layout(h_pad=0.3); fig.savefig(sys.argv[1])
