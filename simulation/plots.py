"""Result figures from results/*.json (run after experiments.py). Usage: python3 plots.py [RESULTS_DIR] [FIG_DIR]. Revised layout: ordered legends,
budget ticks in mJ/slot, panel tags with the operating point."""
import json, sys, numpy as np
from figstyle import plt, POL, OI, COL_W, PAGE_W, errline
R = sys.argv[1] if len(sys.argv) > 1 else 'results'; FIG = (sys.argv[2] if len(sys.argv) > 2 else 'figures') + '/'
for k, v in {'zakeri': 'AoII threshold (Zakeri)', 'gan': 'Pre-identification (Gan)',
             'pehl': 'Age threshold (Pehlivanoğlu)', 'basn': 'Always offload (Basnayaka)'}.items():
    POL[k] = (v,) + POL[k][1:]
ALL = ['cadpp', 'cadpp_blind', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn']
def tag(ax, s, note=None):
    # one centred title per panel, '(a) description', same style as the decision-region figure
    ax.set_title(s + (' ' + note if note else ''), loc='center', fontsize=7.5, pad=3)
from matplotlib.lines import Line2D
BLANK = Line2D([], [], color='none', label=' ')
def ordered_legend(fig, ax0, extra=None, y=1.25):
    hs, ls = ax0.get_legend_handles_labels(); H = dict(zip(ls, hs))
    names = [POL[p][0] for p in ['cadpp', 'cadpp_blind', 'cadpp_label']]
    base = [POL[p][0] for p in ['zakeri', 'gan', 'pehl', 'basn']]
    col = names + [extra if extra else ' ']
    hh, ll = [], []
    for a_, b_ in zip(col, base):          # matplotlib fills legends column by column
        for n in (a_, b_):
            hh.append(H.get(n, BLANK)); ll.append(n)
    fig.legend(hh, ll, loc='upper center', ncol=4, bbox_to_anchor=(0.5, y), columnspacing=1.2)
BT = [0.25, 0.5, 1, 2, 5, 10]
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
def log_yticks(a_, ticks):
    lo, hi = a_.get_ylim(); tk = [v for v in ticks if lo <= v <= hi]
    a_.yaxis.set_major_locator(FixedLocator(tk)); a_.yaxis.set_major_formatter(FixedFormatter([('%g' % v) for v in tk])); a_.yaxis.set_minor_locator(NullLocator())
def budget_ticks(a_):
    a_.set_xticks(BT); a_.set_xticklabels(['0.25', '0.5', '1', '2', '5', '10']); a_.minorticks_off()

# ---------------- Fig. 3: budget ----------------
b = json.load(open(f'{R}/budget.json')); Bs = sorted(b, key=float); x = [float(v)*1e3 for v in Bs]
fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 1.22))
for pol in ALL: errline(ax[0], x, [b[v][pol] for v in Bs], pol)
ax[0].axhline(b[Bs[-1]]['oracle']['aoii'], color='k', ls=':', lw=0.9, label='Genie (knows $X$; %.2f mJ/slot)' % (1e3*b[Bs[-1]]['oracle']['energy']))
ax[0].set(xscale='log', yscale='log', ylabel='average AoII (slots)')
for pol in [p for p in ALL if p != 'basn']: errline(ax[1], x, [b[v][pol] for v in Bs], pol, metric='wrong')
ax[1].set(xscale='log', ylabel='share of slots (%)')
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%g' % round(100*v)))
for pol in ['zakeri', 'gan', 'pehl', 'basn', 'cadpp_blind', 'cadpp_label']:
    lab, c, m, ls = POL[pol]
    y = [-100*b[v][pol]['rel'] for v in Bs]; e = [100*b[v][pol]['rel_ci'] for v in Bs]
    ax[2].errorbar(x, y, yerr=e, color=c, marker=m, ls=ls, mfc='white' if pol.startswith('cadpp_') else c, mec=c, mew=0.8, elinewidth=0.6)
ax[2].axhline(0, color='k', lw=0.6)
ax[2].set(xscale='log', ylabel='AoII reduction (%)', ylim=(-45, 100))
for a, t, nt in zip(ax, '(a) (b) (c)'.split(), ['average AoII', 'slots in which the GS is wrong', 'AoII reduction by CA-DPP']): tag(a, t, nt); budget_ticks(a)
ax[0].set_ylim(0.12, 25); log_yticks(ax[0], [0.2, 0.5, 1, 2, 5, 10, 20])
ordered_legend(fig, ax[0], extra='Genie (knows $X$; %.2f mJ/slot)' % (1e3*b[Bs[-1]]['oracle']['energy']))
fig.tight_layout(w_pad=1.8, rect=(0, 0.045, 1, 1))
fig.text(0.5, 0.0, r'energy budget $\bar E$ (mJ/slot)', ha='center', va='bottom', fontsize=8); fig.savefig(FIG+'fig_budget.pdf'); 

# ---------------- Fig. 4: sensitivity ----------------
fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 1.32), sharey=True)
a = json.load(open(f'{R}/acc.json')); ks = ['on8', 'n2', 'n4', 'n6', 'n8']
ACC = json.load(open(f'{R}/acc_values.json')) if False else None
import numpy as _np
tr = _np.load('traces_sen1floods11.npz'); Xa = _np.concatenate([tr['X_va'], tr['X']]); accx = [100*((_np.concatenate([tr[k+'_va'], tr[k+'_te']]) >= .5) == Xa).mean() for k in ks]
for pol in ALL: errline(ax[0], accx, [a[k][pol] for k in ks], pol)
ax[0].set(yscale='log', xlabel='on-board accuracy (%)', ylabel='average AoII (slots)')
ax[0].set_xticks(accx); ax[0].set_xticklabels(['%.1f' % v for v in accx]); ax[0].minorticks_off()
bl = json.load(open(f'{R}/block.json')); rbs = sorted(bl, key=float); xb = [100*float(r)/(float(r)+0.1) for r in rbs]
for pol in ALL: errline(ax[1], xb, [bl[r][pol] for r in rbs], pol)
ax[1].set(yscale='log', xlabel='share of blocked slots (%)')
pl = json.load(open(f'{R}/payload.json')); Ls = sorted(pl, key=float); xl = [float(L) for L in Ls]
for pol in ['cadpp', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn']: errline(ax[2], xl, [pl[L][pol] for L in Ls], pol)
ax[2].set(xscale='log', yscale='log', xlabel='raw frame size $L_2$ (kB)')
ax[2].set_xticks([12.5, 25, 50, 100, 200]); ax[2].set_xticklabels(['12.5', '25', '50', '100', '200']); ax[2].minorticks_off()
for a_, t, nt in zip(ax, '(a) (b) (c)'.split(), [r'weaker on-board model, $\bar E=0.5$ mJ/slot', r'blockage, $\bar E=0.5$ mJ/slot', r'frame size, $\bar E=2$ mJ/slot']): tag(a_, t, nt)
ax[0].set_ylim(0.25, 25); log_yticks(ax[0], [0.5, 1, 2, 5, 10, 20])
ordered_legend(fig, ax[0], y=1.27)
fig.tight_layout(w_pad=0.8); fig.savefig(FIG+'fig_sensitivity.pdf'); 

# ---------------- Fig. 5: V trade-off and action mix ----------------
V = json.load(open(f'{R}/V.json')); Vs = sorted(V, key=float); xv = [float(v) for v in Vs]
mx = json.load(open(f'{R}/mix.json'))
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 1.4))
# (a) AoII versus V; the top axis gives the energy spent at selected V, the shade marks V within 5% of the budget
en = {float(v): 1e3*V[v]['energy'][0] for v in Vs}
ax[0].axvspan(5e-4, 0.0175, color=OI['grey'], alpha=0.15, lw=0)
ax[0].text(7e-4, 0.215, 'within\nbudget', fontsize=6, color='#555555', va='bottom', linespacing=0.95)
ax[0].errorbar(xv, [V[v]['aoii'][0] for v in Vs], yerr=[V[v]['aoii'][1] for v in Vs], color=OI['verm'], ecolor=OI['verm'], marker='o', ms=3, elinewidth=0.6)
ax[0].set(xscale='log', xlabel='$V$', ylabel='average AoII (slots)', xlim=(6e-4, 5), ylim=(0.2, 1.0))
ax[0].set_xticks([1e-3, 1e-2, 1e-1, 1]); ax[0].minorticks_off()
top = ax[0].twiny(); top.set_xscale('log'); top.set_xlim(ax[0].get_xlim()); top.grid(False); top.minorticks_off()
top.set_xticks([1e-3, 1e-2, 1e-1, 1]); top.set_xticklabels(['%.2f' % en[k] for k in (1e-3, 1e-2, 0.1, 1)], fontsize=6)
# (b) action mix versus distance, side-by-side bars
d = np.array(mx['dist']); m = np.array(mx['mix']); edges = np.linspace(200, 2800, 9); idx = np.clip(np.digitize(d, edges)-1, 0, 7)
n = np.array([m[idx == i].sum() for i in range(8)]); lab = np.array([m[idx == i, 1].sum() for i in range(8)])/n; raw = np.array([m[idx == i, 2].sum() for i in range(8)])/n
ctr = 0.5*(edges[1:]+edges[:-1])/1e3; bw = 0.14
ax[1].bar(ctr-bw/2, 100*lab, width=bw, color=OI['blue'], label='label')
ax[1].bar(ctr+bw/2, 100*raw, width=bw, color=OI['orange'], label='raw upload')
ax[1].set(xlabel='distance to GS (km)', ylabel='share of slots (%)', xlim=(0.15, 2.85)); ax[1].set_xticks([0.5, 1, 1.5, 2, 2.5])
ax[1].legend(loc='upper right', fontsize=6.2, handlelength=1.0, ncol=2, columnspacing=0.8, borderaxespad=0.2); ax[1].set_ylim(0, 10)
top.set_xlabel('energy spent (mJ/slot)', fontsize=6.5, labelpad=2)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.94))
for a_, tt, nt in zip(ax, ['(a)', '(b)'], [r'weight $V$, $\bar E=0.5$ mJ/slot', r'actions, $\bar E=2$ mJ/slot']):
    bb = a_.get_position()
    fig.text((bb.x0+bb.x1)/2, 0.99, tt + ' ' + nt, fontsize=7, va='top', ha='center')
fig.savefig(FIG+'fig_V_mix.pdf'); 
print('ok')
