import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3})
R = 'results2/'; FIG = '../figures/'
ST = {'cadpp': ('CA-DPP (proposed)', '#c0392b', 'o', '-'),
      'cadpp_blind': ('CA-DPP, confidence-blind', '#e59866', 'o', '--'),
      'cadpp_label': ('CA-DPP, label only', '#7d3c98', '^', '-.'),
      'zakeri': ('AoII threshold [Zakeri]', '#2471a3', 's', '-'),
      'gan': ('Pre-identification [Gan]', '#229954', 'D', '-'),
      'pehl': ('Age threshold [Pehlivanoğlu]', '#b7950b', 'v', '-'),
      'basn': ('Always offload [Basnayaka]', '#7f8c8d', 'P', '-')}
e1 = json.load(open(R+'e1_budget.json')); Bs = sorted(e1, key=float)
fig, ax = plt.subplots(figsize=(3.45, 3.1))
for pol, (lab, col, mk, ls) in ST.items():
    ax.plot([float(b)*1e3 for b in Bs], [e1[b][pol]['aoii'] for b in Bs], marker=mk, ls=ls, color=col, label=lab, ms=3.5, lw=1.1)
ax.axhline(e1[Bs[-1]]['oracle']['aoii'], color='k', lw=0.8, ls=':', label='Genie (knows $X$)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'energy budget $\bar E$ (mJ/slot)'); ax.set_ylabel('average AoII (slots)')
ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=2, columnspacing=0.8, handlelength=2.2, frameon=False)
ax.set_ylim(0.15, 20)
fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.36); fig.savefig(FIG+'aoii_budget.pdf')
e2 = json.load(open(R+'e2_payload.json')); e3 = json.load(open(R+'e3_V.json'))
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.05))
Ls = sorted(e2, key=float)
for pol in ('cadpp', 'cadpp_label', 'zakeri'):
    lab, col, mk, ls = ST[pol]
    ax[0].plot([float(l) for l in Ls], [e2[l][pol]['aoii'] for l in Ls], marker=mk, ls=ls, color=col, ms=3, lw=1.1, label={'cadpp':'CA-DPP','cadpp_label':'CA-DPP, label only','zakeri':'AoII threshold [Zakeri]'}[pol])
ax[0].set_xscale('log'); ax[0].set_xticks([12.5, 50, 200]); ax[0].set_xticklabels(['12.5', '50', '200'])
ax[0].minorticks_off(); ax[0].set_ylim(0.235, 0.29)
ax[0].set_xlabel('raw payload $L_2$ (kB)'); ax[0].set_ylabel('average AoII (slots)')
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=6, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
Vs = sorted(e3, key=float)
ax[1].plot([float(v) for v in Vs], [e3[v]['aoii'] for v in Vs], 'o-', color='#c0392b', ms=3)
ax[1].set_xscale('log'); ax[1].set_xlabel('$V$'); ax[1].set_ylabel('AoII', color='#c0392b')
a2 = ax[1].twinx(); a2.plot([float(v) for v in Vs], [e3[v]['energy']*1e3 for v in Vs], 's--', color='#2471a3', ms=3)
a2.axhline(0.5, color='#2471a3', lw=0.6, ls=':'); a2.set_ylabel('energy (mJ/slot)', color='#2471a3'); a2.grid(False)
fig.subplots_adjust(left=0.16, right=0.86, bottom=0.22, top=0.86, wspace=0.6)
fig.savefig(FIG+'payload_V.pdf')
print('ok')
