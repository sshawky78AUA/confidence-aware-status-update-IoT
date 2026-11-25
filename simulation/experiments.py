"""Experiments for the paper (trace-driven, Sen1Floods11), chip-level cross-validation.
Protocol: the 174 validation + test chips are split into 5 folds (fixed seed). For each fold,
every policy parameter (V and the belief model for CA-DPP and its ablations, thresholds for
the baselines) and every estimated model quantity (ground error per confidence bin, label
precision) is chosen on the other 4 folds (2 runs of one pass each); the chosen value is then
evaluated on the held-out fold over 4 runs of 4 passes (chip order, fading, blockage).
This gives 20 held-out runs per policy. Runs share random numbers across policies, so paired
statistics are used for the differences. Usage: python3 experiments.py OUTDIR [what]"""
import sim, numpy as np, json, os, sys
from multiprocessing import Pool
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'; os.makedirs(OUT, exist_ok=True)
K = 5; TUNE_SEEDS = (101, 102); TEST_SEEDS = (1, 2, 3, 4)
CH = np.random.default_rng(0).permutation(np.unique(sim.TR['chip_all']))
FOLDS = [tuple(int(c) for c in CH[f::K]) for f in range(K)]
def split(f, part):
    ev = FOLDS[f]; tu = tuple(c for g in range(K) if g != f for c in FOLDS[g])
    return ('cv', tu, tu, 1) if part == 'tune' else ('cv', ev, tu, 4)
def VG(B):                                # V window per budget (the virtual queue scales with 1/budget)
    return [0.001, 0.003, 0.01, 0.03] if B <= 0.0005 else ([0.03, 0.1, 0.3, 1] if B <= 0.002 else [0.3, 1, 3, 10])
BEL = [0, 0.25, 0.5]                      # belief: 0 symmetric update; else prior-corrected with discount nu
BETA = [0]                                # look-ahead discount; 0.5 and 0.8 were tried and did not help (archive/results_beta)
GRIDS = {'gan': [None], 'oracle': [None],
         'zakeri': [1, 2, 3, 5], 'pehl': [(A, q) for A in (1, 3, 10) for q in (0.5, 0.9, 1.1)],
         'basn': [1, 2, 5, 10, 50]}
POLS = ['cadpp', 'cadpp_blind', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn', 'oracle']

def job(a):
    pol, B, par, over, s, f, part = a
    P = dict(sim.P); P.update({k: v for k, v in over.items() if not k.startswith('_')})
    r = sim.run(pol, B, par, s, P=P, mixlog=over.get('_mix', False), split=split(f, part))
    return (pol, B, json.dumps(par), json.dumps(over), s, f, part), r

def run_jobs(jobs):
    with Pool(4) as p: return dict(p.map(job, jobs, chunksize=2))

def ci95(x):
    x = np.asarray(x, float); m = x.mean(); h = stats.t.ppf(0.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(m), float(h)

def grid(pol, B):
    if pol.startswith('cadpp'): return [(v, g, be) for v in VG(B) for g in BEL for be in BETA]
    return GRIDS[pol]

def tune(points, pols):
    tj = [(pol, B, par, over, s, f, 'tune') for (_, B, over) in points for pol in pols for par in grid(pol, B) for s in TUNE_SEEDS for f in range(K)]
    TR = run_jobs(tj); chosen = {}
    for lab, B, over in points:
        for pol in pols:
            for f in range(K):
                best = None
                for par in grid(pol, B):
                    rr = [TR[(pol, B, json.dumps(par), json.dumps(over), s, f, 'tune')] for s in TUNE_SEEDS]
                    a = np.mean([r['aoii'] for r in rr]); e = np.mean([r['energy'] for r in rr]); feas = e <= 1.05*B or pol == 'oracle'
                    key = (0 if feas else 1, a if feas else e)
                    if best is None or key < best[0]: best = (key, par)
                chosen[(lab, pol, f)] = best[1]
    return chosen

def sweep(points, pols=POLS):
    chosen = tune(points, pols)
    ej = [(pol, B, chosen[(lab, pol, f)], over, s, f, 'eval') for (lab, B, over) in points for pol in pols for f in range(K) for s in TEST_SEEDS]
    ER = run_jobs(ej); out = {}
    for lab, B, over in points:
        out[lab] = {}
        R = {pol: [ER[(pol, B, json.dumps(chosen[(lab, pol, f)]), json.dumps(over), s, f, 'eval')] for f in range(K) for s in TEST_SEEDS] for pol in pols}
        base = np.array([r['aoii'] for r in R['cadpp']])
        for pol in pols:
            a = np.array([r['aoii'] for r in R[pol]]); d = {}
            for m in ('aoii', 'wrong', 'energy', 'p95', 'p99'):
                d[m], d[m+'_ci'] = ci95([r[m] for r in R[pol]])
            d['frac'] = np.mean([r['frac'] for r in R[pol]], axis=0).tolist()
            d['par'] = [json.dumps(chosen[(lab, pol, f)]) for f in range(K)]
            d['runs'] = a.tolist()
            if pol != 'cadpp':
                rel = (base - a)/a
                d['rel'], d['rel_ci'] = ci95(rel)
                diff = base - a
                d['p_t'] = float(stats.ttest_rel(base, a).pvalue) if np.any(diff != 0) else 1.0
                d['p_w'] = float(stats.wilcoxon(base, a).pvalue) if np.any(diff != 0) else 1.0
                fm = (base - a).reshape(K, -1).mean(1); d['folds_better'] = int((fm < 0).sum())
            if over.get('_mix'):
                d['mix'] = np.sum([r['mix'] for r in R[pol]], axis=0).tolist()
            out[lab][pol] = d
    return out, chosen

if __name__ == '__main__':
    what = sys.argv[2].split(',') if len(sys.argv) > 2 else ['budget', 'acc', 'block', 'payload', 'V', 'mix']
    if 'budget' in what:
        r, ch = sweep([(str(B), B, {}) for B in (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01)])
        json.dump(r, open(f'{OUT}/budget.json', 'w')); print('budget', flush=True)
        json.dump({str(f): ch[('0.0005', 'cadpp', f)] for f in range(K)}, open(f'{OUT}/chosen_05.json', 'w'))
    if 'acc' in what:
        r, _ = sweep([(k, 0.0005, {'onboard': k}) for k in ('on8', 'n2', 'n4', 'n6', 'n8')])
        json.dump(r, open(f'{OUT}/acc.json', 'w')); print('acc', flush=True)
    if 'block' in what:
        r, _ = sweep([(str(rb), 0.0005, {'p_block': rb}) for rb in (0.0, 0.01, 0.02, 0.05, 0.1)])
        json.dump(r, open(f'{OUT}/block.json', 'w')); print('block', flush=True)
    if 'payload' in what:
        r, _ = sweep([(str(L), 0.002, {'L2': L*8e3}) for L in (12.5, 25, 50, 100, 200)],
                     ['cadpp', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn'])
        json.dump(r, open(f'{OUT}/payload.json', 'w')); print('payload', flush=True)
    if 'V' in what:
        ch = json.load(open(f'{OUT}/chosen_05.json'))
        Vs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        R = run_jobs([('cadpp', 0.0005, (v,) + tuple(ch[str(f)][1:]), {}, s, f, 'eval') for v in Vs for f in range(K) for s in TEST_SEEDS])
        out = {}
        for v in Vs:
            rr = [R[('cadpp', 0.0005, json.dumps((v,) + tuple(ch[str(f)][1:])), json.dumps({}), s, f, 'eval')] for f in range(K) for s in TEST_SEEDS]
            out[str(v)] = {m: ci95([r[m] for r in rr]) for m in ('aoii', 'energy', 'Zend')}
        json.dump(out, open(f'{OUT}/V.json', 'w')); print('V', flush=True)
    if 'mix' in what:
        r, _ = sweep([('mix', 0.002, {'_mix': True})], ['cadpp'])
        LT = sim.link_tables(sim.P)
        json.dump(dict(dist=LT['dist'].tolist(), q2=LT['q2'].tolist(), E2=LT['E2'].tolist(), mix=r['mix']['cadpp']['mix']), open(f'{OUT}/mix.json', 'w'))
    print('done', flush=True)
