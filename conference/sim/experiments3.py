"""Experiments for the paper (trace-driven, Sen1Floods11).
Protocol: every policy parameter (V for CA-DPP and its ablations, thresholds for the
baselines) is tuned on the VALIDATION chips (2 runs); the chosen value is then evaluated
on the TEST chips over 10 independent runs (chip order, fading, blockage). Runs share
random numbers across policies, so paired statistics are used for the differences.
Writes results3/*.json. Usage: python3 experiments3.py [OUTDIR]"""
import sim3, numpy as np, json, os, sys
from multiprocessing import Pool
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results3'; os.makedirs(OUT, exist_ok=True)
TUNE_SEEDS = (101, 102); TEST_SEEDS = tuple(range(1, 11))
VG = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
GG = [0.25, 0.5, 1.0]                      # evidence discount gamma
VGG = [(v, g) for v in VG for g in GG]
GRIDS = {'cadpp': VGG, 'cadpp_blind': VGG, 'cadpp_label': VGG, 'gan': [None], 'oracle': [None],
         'zakeri': [1, 2, 3, 5], 'pehl': [(A, q) for A in (1, 3, 10) for q in (0.5, 0.9, 1.1)],
         'basn': [1, 2, 5, 10, 50]}
POLS = ['cadpp', 'cadpp_blind', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn', 'oracle']

def job(a):
    pol, B, par, over, s, split = a
    P = dict(sim3.P); P.update({k: v for k, v in over.items() if not k.startswith('_')})
    r = sim3.run(pol, B, par, s, P=P, mixlog=over.get('_mix', False), split=split)
    r.pop('trace', None); return (pol, B, json.dumps(par), json.dumps(over), s, split), r

def run_jobs(jobs):
    with Pool(4) as p: return dict(p.map(job, jobs, chunksize=1))

def ci95(x):
    x = np.asarray(x, float); m = x.mean(); h = stats.t.ppf(0.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(m), float(h)

def sweep(points, pols=POLS):
    # 1) tuning on validation
    tj = [(pol, B, par, over, s, 'va') for (_, B, over) in points for pol in pols for par in GRIDS[pol] for s in TUNE_SEEDS]
    TR = run_jobs(tj); chosen = {}
    for lab, B, over in points:
        for pol in pols:
            best = None
            for par in GRIDS[pol]:
                rr = [TR[(pol, B, json.dumps(par), json.dumps(over), s, 'va')] for s in TUNE_SEEDS]
                a = np.mean([r['aoii'] for r in rr]); e = np.mean([r['energy'] for r in rr]); feas = e <= 1.05*B or pol in ('oracle',)
                key = (0 if feas else 1, a if feas else e)
                if best is None or key < best[0]: best = (key, par)
            chosen[(lab, pol)] = best[1]
    # 2) evaluation on test
    ej = [(pol, B, chosen[(lab, pol)], over, s, 'te') for (lab, B, over) in points for pol in pols for s in TEST_SEEDS]
    ER = run_jobs(ej); out = {}
    for lab, B, over in points:
        out[lab] = {}; R = {pol: [ER[(pol, B, json.dumps(chosen[(lab, pol)]), json.dumps(over), s, 'te')] for s in TEST_SEEDS] for pol in pols}
        base = np.array([r['aoii'] for r in R['cadpp']])
        for pol in pols:
            a = np.array([r['aoii'] for r in R[pol]]); d = {}
            for m in ('aoii', 'wrong', 'energy', 'p95', 'p99'):
                d[m], d[m+'_ci'] = ci95([r[m] for r in R[pol]])
            d['frac'] = np.mean([r['frac'] for r in R[pol]], axis=0).tolist(); d['par'] = json.dumps(chosen[(lab, pol)])
            d['runs'] = a.tolist()
            if pol != 'cadpp':
                rel = (base - a)/a              # relative change of CA-DPP w.r.t. this policy, per run
                d['rel'], d['rel_ci'] = ci95(rel)
                diff = base - a
                d['p_t'] = float(stats.ttest_rel(base, a).pvalue) if np.any(diff != 0) else 1.0
                d['p_w'] = float(stats.wilcoxon(base, a).pvalue) if np.any(diff != 0) else 1.0
            if over.get('_mix'):
                d['mix'] = np.sum([r['mix'] for r in R[pol]], axis=0).tolist()
            out[lab][pol] = d
    return out

if __name__ == '__main__':
    what = sys.argv[2].split(',') if len(sys.argv) > 2 else ['budget', 'acc', 'block', 'payload', 'V', 'mix']
    if 'budget' in what:
        r = sweep([(str(B), B, {}) for B in (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01)])
        json.dump(r, open(f'{OUT}/budget.json', 'w')); print('budget', flush=True)
    if 'acc' in what:
        r = sweep([(k, 0.0005, {'onboard': k}) for k in ('on8', 'n2', 'n4', 'n6', 'n8')])
        json.dump(r, open(f'{OUT}/acc.json', 'w')); print('acc', flush=True)
    if 'block' in what:
        r = sweep([(str(rb), 0.0005, {'p_block': rb}) for rb in (0.0, 0.01, 0.02, 0.05, 0.1)])
        json.dump(r, open(f'{OUT}/block.json', 'w')); print('block', flush=True)
    if 'payload' in what:
        r = sweep([(str(L), 0.002, {'L2': L*8e3}) for L in (12.5, 25, 50, 100, 200)],
                  ['cadpp', 'cadpp_label', 'zakeri', 'gan', 'pehl', 'basn'])
        json.dump(r, open(f'{OUT}/payload.json', 'w')); print('payload', flush=True)
    if 'V' in what:
        Vs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        g = json.loads(json.load(open(f'{OUT}/budget.json'))['0.0005']['cadpp']['par'])[1]
        R = run_jobs([('cadpp', 0.0005, (v, g), {}, s, 'te') for v in Vs for s in TEST_SEEDS])
        out = {}
        for v in Vs:
            rr = [R[('cadpp', 0.0005, json.dumps((v, g)), json.dumps({}), s, 'te')] for s in TEST_SEEDS]
            out[str(v)] = {m: ci95([r[m] for r in rr]) for m in ('aoii', 'energy', 'Zend')}
        json.dump(out, open(f'{OUT}/V.json', 'w')); print('V', flush=True)
    if 'mix' in what:
        r = sweep([('mix', 0.002, {'_mix': True})], ['cadpp'])
        LT = sim3.link_tables(sim3.P)
        json.dump(dict(dist=LT['dist'].tolist(), q2=LT['q2'].tolist(), E2=LT['E2'].tolist(), mix=r['mix']['cadpp']['mix']), open(f'{OUT}/mix.json', 'w'))
    print('done', flush=True)
