"""Shared IEEE figure style: Times-like fonts at 8 pt, vector PDF with embedded TrueType fonts,
Okabe-Ito colorblind-safe palette, one marker and line style per policy."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL_W, PAGE_W = 3.5, 7.16    # IEEE column and page widths (inches)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'lines.markersize': 3.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.top': True, 'ytick.right': True,
    'axes.grid': True, 'grid.linewidth': 0.4, 'grid.alpha': 0.35, 'grid.linestyle': '-',
    'legend.frameon': False, 'legend.handlelength': 2.0, 'legend.columnspacing': 1.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'errorbar.capsize': 1.8})
OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
      'yellow': '#F0E442', 'blue': '#0072B2', 'verm': '#D55E00', 'purple': '#CC79A7', 'grey': '#888888'}
POL = {  # label, color, marker, linestyle
    'cadpp':       ('CA-DPP (proposed)', OI['verm'], 'o', '-'),
    'cadpp_blind': ('CA-DPP, confidence-blind', OI['orange'], 'o', (0, (4, 1.5))),
    'cadpp_label': ('CA-DPP, label only', OI['purple'], '^', (0, (4, 1.5))),
    'zakeri':      ('AoII threshold, after [4]', OI['blue'], 's', '-'),
    'gan':         ('Pre-identification, after [7]', OI['green'], 'D', '-'),
    'pehl':        ('Age threshold, after [8]', OI['sky'], 'v', '-'),
    'basn':        ('Always offload, after [9]', OI['grey'], 'P', '-'),
    'oracle':      ('Genie (knows $X$)', OI['black'], None, ':'),
}
def errline(ax, x, d, pol, metric='aoii', **kw):
    lab, c, m, ls = POL[pol]
    y = [v[metric] for v in d]; e = [v[metric+'_ci'] for v in d]
    return ax.errorbar(x, y, yerr=e, color=c, marker=m, ls=ls, label=lab, mfc='white' if pol.startswith('cadpp_') else c, mec=c, mew=0.8, elinewidth=0.6, **kw)
