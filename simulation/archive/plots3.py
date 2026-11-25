"""Figures 3-5 from results3/*.json (run after experiments3.py)."""
import json, sys, numpy as np
from figstyle import plt, POL, OI, COL_W, PAGE_W, errline
R = sys.argv[1] if len(sys.argv) > 1 else 'results3'; FIG = '../figures/'
for k, v in {'zakeri': 'AoII threshold (Zakeri)', 'gan': 'Pre-identification (Gan)',
             'pehl': 'Age threshold (Pehlivanoğlu)', 'basn': 'Always offload (Basnayaka)'}.items():
    POL[k] = (v,) + POL[k][1:]
ALL = ['cadpp', 'cadpp_blind', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn']
def tag(ax, s): ax.set_title(s, loc='left', fontsize=8, fontweight='bold', pad=3)

# ---------------- Fig. 3: budget ----------------
b = json.load(open(f'{R}/budget.json')); Bs = sorted(b, key=float); x = [float(v)*1e3 for v in Bs]
fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 1.9))
for pol in ALL: errline(ax[0], x, [b[v][pol] for v in Bs], pol)
ax[0].axhline(b[Bs[-1]]['oracle']['aoii'], color='k', ls=':', lw=0.9, label='genie (knows $X$, %.2f mJ/slot)' % (1e3*b[Bs[-1]]['oracle']['energy']))
ax[0].set(xscale='log', yscale='log', xlabel=r'energy budget $\bar E$ (mJ/slot)', ylabel='average AoII (slots)')
for pol in [p for p in ALL if p != 'basn']: errline(ax[1], x, [b[v][pol] for v in Bs], pol, metric='wrong')
ax[1].set(xscale='log', xlabel=r'energy budget $\bar E$ (mJ/slot)', ylabel=r'fraction of slots GS is wrong')
for pol in ['zakeri', 'gan', 'pehl', 'basn', 'cadpp_blind', 'cadpp_label']:
    lab, c, m, ls = POL[pol]
    y = [-100*b[v][pol]['rel'] for v in Bs]; e = [100*b[v][pol]['rel_ci'] for v in Bs]
    ax[2].errorbar(x, y, yerr=e, color=c, marker=m, ls=ls, mfc='white' if pol.startswith('cadpp_') else c, mec=c, mew=0.8, elinewidth=0.6)
ax[2].axhline(0, color='k', lw=0.6)
ax[2].set(xscale='log', xlabel=r'energy budget $\bar E$ (mJ/slot)', ylabel='AoII reduction by CA-DPP (%)', ylim=(-45, 100))
for a, t in zip(ax, '(a) (b) (c)'.split()): tag(a, t)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(w_pad=1.2); fig.savefig(FIG+'fig_budget.pdf'); fig.savefig('/tmp/claude-0/fig_budget.png', dpi=160)

# ---------------- Fig. 4: sensitivity ----------------
fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 1.85))
a = json.load(open(f'{R}/acc.json')); ks = ['on8', 'n2', 'n4', 'n6', 'n8']
ACC = json.load(open(f'{R}/acc_values.json')) if False else None
import numpy as _np
tr = _np.load('traces_sen1floods11.npz'); Xa = _np.concatenate([tr['X_va'], tr['X']]); accx = [100*((_np.concatenate([tr[k+'_va'], tr[k+'_te']]) >= .5) == Xa).mean() for k in ks]
for pol in ALL: errline(ax[0], accx, [a[k][pol] for k in ks], pol)
ax[0].set(yscale='log', xlabel='on-board accuracy (%)', ylabel='average AoII (slots)')
bl = json.load(open(f'{R}/block.json')); rbs = sorted(bl, key=float); xb = [100*float(r)/(float(r)+0.1) for r in rbs]
for pol in ALL: errline(ax[1], xb, [bl[r][pol] for r in rbs], pol)
ax[1].set(yscale='log', xlabel='share of blocked slots (%)', ylabel='average AoII (slots)')
pl = json.load(open(f'{R}/payload.json')); Ls = sorted(pl, key=float); xl = [float(L) for L in Ls]
for pol in ['cadpp', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn']: errline(ax[2], xl, [pl[L][pol] for L in Ls], pol)
ax[2].set(xscale='log', yscale='log', xlabel='raw payload $L_2$ (kB)', ylabel='average AoII (slots)')
ax[2].set_xticks([12.5, 25, 50, 100, 200]); ax[2].set_xticklabels(['12.5', '25', '50', '100', '200']); ax[2].minorticks_off()
for a_, t in zip(ax, '(a) (b) (c)'.split()): tag(a_, t)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(w_pad=1.2); fig.savefig(FIG+'fig_sensitivity.pdf'); fig.savefig('/tmp/claude-0/fig_sens.png', dpi=160)

# ---------------- Fig. 5: V trade-off and action mix ----------------
V = json.load(open(f'{R}/V.json')); Vs = sorted(V, key=float); xv = [float(v) for v in Vs]
mx = json.load(open(f'{R}/mix.json'))
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 1.55))
ax[0].errorbar(xv, [V[v]['aoii'][0] for v in Vs], yerr=[V[v]['aoii'][1] for v in Vs], color=OI['verm'], marker='o', label='AoII')
ax[0].set(xscale='log', xlabel='$V$', ylabel='average AoII (slots)')
t = ax[0].twinx(); t.grid(False)
t.errorbar(xv, [1e3*V[v]['energy'][0] for v in Vs], yerr=[1e3*V[v]['energy'][1] for v in Vs], color=OI['blue'], marker='s', ls=(0, (4, 1.5)), mfc='white', mec=OI['blue'])
t.axhline(0.5, color=OI['blue'], lw=0.6, ls=':'); t.set_ylabel('energy (mJ/slot)', color=OI['blue']); t.tick_params(axis='y', colors=OI['blue'])
ax[0].yaxis.label.set_color(OI['verm']); ax[0].tick_params(axis='y', colors=OI['verm'])
d = np.array(mx['dist']); m = np.array(mx['mix']); edges = np.linspace(200, 2800, 9); idx = np.clip(np.digitize(d, edges)-1, 0, 7)
n = np.array([m[idx == i].sum() for i in range(8)]); lab = np.array([m[idx == i, 1].sum() for i in range(8)])/n; raw = np.array([m[idx == i, 2].sum() for i in range(8)])/n
ctr = 0.5*(edges[1:]+edges[:-1])/1e3
ax[1].bar(ctr, 100*raw, width=0.28, color=OI['orange'], label='raw'); ax[1].bar(ctr, 100*lab, width=0.28, bottom=100*raw, color=OI['blue'], label='label')
ax[1].set(xlabel='distance to GS (km)', ylabel='slots with a transmission (%)'); ax[1].legend(loc='upper right', fontsize=6.5)
for a_, tt in zip(ax, '(a) (b)'.split()): tag(a_, tt)
fig.tight_layout(w_pad=0.6); fig.savefig(FIG+'fig_V_mix.pdf'); fig.savefig('/tmp/claude-0/fig_vmix.png', dpi=160)
print('ok')
