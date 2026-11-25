"""All experiments of the paper (trace-driven). Writes results2/*.json. About 10 min on 4 cores."""
import sim2, numpy as np, json, os
from multiprocessing import Pool
OUT = '/tmp/claude-0/results2'; os.makedirs(OUT, exist_ok=True)
SEEDS = (1, 2, 3); V = 3.0
VG = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
GRIDS = {'cadpp': VG, 'cadpp_blind': VG, 'cadpp_label': VG, 'gan': [None], 'oracle': [None],
         'zakeri': [1, 2, 3, 5], 'pehl': [(A, q) for A in (1, 3, 10) for q in (0.5, 0.9, 1.1)],
         'basn': [1, 2, 5, 10, 50]}

def job(a):
    pol, B, par, over, s = a
    P = dict(sim2.P); P.update(over)
    r = sim2.run(pol, B, par, s, P=P, mixlog=over.get('_mix', False))
    return pol, B, str(par), json.dumps(over), s, r

def sweep(points, pols):
    jobs = [(pol, B, par, over, s) for (lab, B, over) in points for pol in pols for par in GRIDS[pol] for s in SEEDS]
    with Pool(4) as p: R = p.map(job, jobs, chunksize=2)
    out = {}
    for lab, B, over in points:
        out[lab] = {}
        for pol in pols:
            best = None
            for par in GRIDS[pol]:
                rr = [r for (pp, b, pa, ov, s, r) in R if pp == pol and b == B and pa == str(par) and ov == json.dumps(over)]
                m = dict(aoii=float(np.mean([r['aoii'] for r in rr])), aoii_sd=float(np.std([r['aoii'] for r in rr])),
                         wrong=float(np.mean([r['wrong'] for r in rr])), energy=float(np.mean([r['energy'] for r in rr])),
                         frac=np.mean([r['frac'] for r in rr], axis=0).tolist(), par=str(par),
                         Zend=float(np.mean([r['Zend'] for r in rr])))
                if rr[0]['mix'] is not None: m['mix'] = np.sum([r['mix'] for r in rr], axis=0).tolist()
                feas = m['energy'] <= 1.05*B; m['feas'] = feas
                if best is None or (feas and (not best['feas'] or m['aoii'] < best['aoii'])) or (not best['feas'] and not feas and m['energy'] < best['energy']):
                    best = m
            out[lab][pol] = best
    return out

if __name__ == '__main__':
    ALL = list(GRIDS)
    e1 = sweep([(str(B), B, {}) for B in (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01)], ALL)
    json.dump(e1, open(OUT+'/e1_budget.json', 'w'), indent=1); print('e1')
    e2 = sweep([(str(L), 0.002, {'L2': L*8e3}) for L in (12.5, 25, 50, 100, 200)],
               ['cadpp', 'cadpp_label', 'gan', 'pehl', 'basn', 'zakeri'])
    json.dump(e2, open(OUT+'/e2_payload.json', 'w'), indent=1); print('e2')
    GRIDS['cadpp'] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
    jobs = [('cadpp', 0.0005, v, {}, s) for v in GRIDS['cadpp'] for s in SEEDS]
    with Pool(4) as p: R = p.map(job, jobs)
    e3 = {str(v): dict(aoii=float(np.mean([r['aoii'] for (pp, b, pa, ov, s, r) in R if pa == str(v)])),
                       energy=float(np.mean([r['energy'] for (pp, b, pa, ov, s, r) in R if pa == str(v)])),
                       Zend=float(np.mean([r['Zend'] for (pp, b, pa, ov, s, r) in R if pa == str(v)])))
          for v in GRIDS['cadpp']}
    json.dump(e3, open(OUT+'/e3_V.json', 'w'), indent=1); print('e3')
    GRIDS['cadpp'] = [0.1]
    e4 = sweep([('mix', 0.002, {'_mix': True})], ['cadpp'])
    LT = sim2.link_tables(sim2.P)
    json.dump(dict(dist=LT['dist'].tolist(), mix=e4['mix']['cadpp']['mix']), open(OUT+'/e4_mix.json', 'w')); print('done')
