"""Table III from results/budget.json. Usage: python3 table.py [RESULTS_DIR] [OUT.tex]: mean and 95% CI over 20 held-out runs (5-fold chip CV), paired comparison."""
import json, sys
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
b = json.load(open(f'{R}/budget.json'))
NAMES = [('cadpp', 'CA-DPP (proposed)'), ('zakeri', 'AoII threshold~\\cite{zakeri2023aoiieh}'), ('gan', 'Pre-identification~\\cite{gan2025taoi}'),
         ('pehl', 'Age threshold~\\cite{pehlivanoglu2026orbit}'), ('basn', 'Always offload~\\cite{basnayaka2025aomi}'),
         ('cadpp_blind', 'CA-DPP, confidence-blind'), ('cadpp_label', 'CA-DPP, label only')]
def fmt(v, h, d=3): return f'${v:.{d}f}\\pm{h:.{d}f}$'
def pfmt(p): return '$<\\!10^{-3}$' if p < 1e-3 else f'${p:.3f}$'
L = ['\\begin{table*}[t]', '\\caption{Trace-driven results at two energy budgets (mean $\\pm$ 95\\% confidence half-width over 20 held-out runs: 5-fold cross-validation over chips, 4 runs per fold). '
     '$\\Delta$: change in average AoII obtained by CA-DPP relative to the policy in that row (negative means CA-DPP is lower), with paired 95\\% confidence interval, paired $t$-test $p$-value, and the number of folds (out of 5) in which CA-DPP has the lower mean.}',
     '\\label{tab:results}', '\\centering\\footnotesize\\setlength{\\tabcolsep}{3.2pt}',
     '\\begin{tabular}{@{}l' + 'cccc' * 2 + '@{}}', '\\toprule',
     ' & \\multicolumn{4}{c}{$\\bar E=0.25$\\,mJ/slot} & \\multicolumn{4}{c}{$\\bar E=2$\\,mJ/slot}\\\\',
     '\\cmidrule(lr){2-5}\\cmidrule(l){6-9}',
     'Policy & AoII & $\\Delta$ (\\%) & $p$ & folds & AoII & $\\Delta$ (\\%) & $p$ & folds\\\\', '\\midrule']
for k, n in NAMES:
    row = [n]
    for B in ('0.00025', '0.002'):
        x = b[B][k]
        row += [fmt(x['aoii'], x['aoii_ci'])]
        if k == 'cadpp': row += ['--', '--', '--']
        else: row += [f"${100*x['rel']:+.1f}\\pm{100*x['rel_ci']:.1f}$", pfmt(x['p_t']), f"{x['folds_better']}/5"]
    L.append(' & '.join(row) + ' \\\\')
    if k == 'basn': L.append('\\midrule')
o = b['0.01']['oracle']
L += ['\\midrule', f"Genie (knows $X$) & \\multicolumn{{8}}{{l}}{{AoII ${o['aoii']:.3f}\\pm{o['aoii_ci']:.3f}$, $\\Prob$(GS wrong) ${o['wrong']:.3f}$, energy ${1e3*o['energy']:.2f}$\\,mJ/slot when unconstrained}} \\\\",
      '\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(sys.argv[2] if len(sys.argv) > 2 else 'table_results.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
