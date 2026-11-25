"""Robustness to temporally correlated classifier errors, and the choice of the discount nu.
Usage: python3 robust_corr.py OUTDIR [synthetic,nu_cv]

(1) synthetic: X(t) is the Markov chain of the data. The on-board error indicator w(t) is a two-state Markov
    chain with the measured error rate and persistence pi = P{w(t+1)=1 | w(t)=1}, from pi = error rate
    (independent errors, Assumption 1 holds) to 0.8 (the data has 0.56). Given X and w, the confidence and the
    ground output are drawn jointly from data slots with the same X and the same on-board correctness, so the
    classifier stays calibrated. CA-DPP runs with nu = 1, nu = 0.25 and the rule nu = (1-rho)/(1+rho), where
    rho = (pi - e)/(1 - e) is the lag-one correlation of w; V and the AoII-threshold k are tuned per setting.
(2) nu_cv: on the real traces, CA-DPP with nu fixed to each value under the cross-validation protocol of
    experiments.py, to show how sensitive the results are to nu."""
import numpy as np, json, os, sys
import sim, optgap
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'; os.makedirs(OUT, exist_ok=True)
TR = sim.TR; X_all = optgap.X_all; corr = optgap.corr; pon = optgap.pon; pgr = optgap.pgr; pflip = optgap.pflip
E_RATE = 1 - corr.mean()

def synth_corr(seed, T, pi):
    rng = np.random.default_rng(seed)
    X, _, _ = optgap.synth(seed, T)
    w = np.zeros(T, int); w[0] = rng.random() < E_RATE
    p_on = E_RATE*(1-pi)/(1-E_RATE)                 # P{w=1 | w=0} that keeps the error rate
    u = rng.random(T)
    for t in range(1, T): w[t] = (u[t] < pi) if w[t-1] else (u[t] < p_on)
    idx = np.empty(T, int)
    for x in (0, 1):
        for r in (0, 1):
            pool = np.where((X_all == x) & (corr == r))[0]; m = (X == x) & (w == 1-r)
            idx[m] = rng.choice(pool, m.sum())
    return X, pon[idx], pgr[idx]

def run_corr(pol, B, par, seed, pi, T=60000):
    X, po, pg = synth_corr(seed, T, pi)
    TR['X_syn'], TR['on8_syn'], TR['gr_syn'] = X, po, pg
    old = sim.make_trace
    sim.make_trace = lambda s, reps=4, split='te': (np.arange(T), np.random.default_rng(seed+7))
    try: r = sim.run(pol, B, par, seed, P=dict(optgap.P), split='syn')
    finally: sim.make_trace = old
    return r

def job(a):
    pol, B, par, s, pi = a; r = run_corr(pol, B, par, s, pi); return (pol, B, json.dumps(par), s, pi), (r['aoii'], r['energy'])

def rule(pi): rho = (pi - E_RATE)/(1 - E_RATE); return float((1-rho)/(1+rho))

if __name__ == '__main__':
    from multiprocessing import Pool
    what = sys.argv[2].split(',') if len(sys.argv) > 2 else ['synthetic', 'nu_cv']
    NP = int(os.environ.get('NPROC', 4))
    if 'synthetic' in what:
        PIS = [round(E_RATE, 3), 0.3, 0.56, 0.8]; budgets = [0.0005, 0.002]
        Vg = {0.0005: [0.003, 0.01, 0.03, 0.1], 0.002: [0.1, 0.3, 1, 3]}
        def nus(pi): return sorted(set([1.0, 0.25, round(rule(pi), 3)]))
        tune = [('cadpp', B, (v, nu, 0), s, pi) for pi in PIS for B in budgets for nu in nus(pi) for v in Vg[B] for s in (11, 12)]
        tune += [('zakeri', B, k, s, pi) for pi in PIS for B in budgets for k in (1, 2, 3, 5) for s in (11, 12)]
        with Pool(NP) as p: TU = dict(p.map(job, tune))
        def pick(pol, B, grid, pi):
            best = None
            for par in grid:
                rr = [TU[(pol, B, json.dumps(par), s, pi)] for s in (11, 12)]; a = np.mean([r[0] for r in rr]); e = np.mean([r[1] for r in rr])
                key = (0 if e <= 1.05*B else 1, a if e <= 1.05*B else e)
                if best is None or key < best[0]: best = (key, par)
            return best[1]
        ch = {}
        for pi in PIS:
            for B in budgets:
                for nu in nus(pi): ch[(pi, B, 'cadpp', nu)] = pick('cadpp', B, [(v, nu, 0) for v in Vg[B]], pi)
                ch[(pi, B, 'zakeri', None)] = pick('zakeri', B, [1, 2, 3, 5], pi)
        SE = (1, 2, 3, 4, 5, 6)
        ev = [(key[2], key[1], par, s, key[0]) for key, par in ch.items() for s in SE]
        with Pool(NP) as p: EV = dict(p.map(job, ev))
        out = {}
        for (pi, B, pol, nu), par in ch.items():
            rr = [EV[(pol, B, json.dumps(par), s, pi)] for s in SE]
            a = np.array([r[0] for r in rr])
            out.setdefault(str(pi), {}).setdefault(str(B), {})[pol if nu is None else f'cadpp_nu{nu}'] = dict(
                par=par, aoii=float(a.mean()), sd=float(a.std(ddof=1)), energy=float(np.mean([r[1] for r in rr])), runs=a.tolist())
            print(pi, B, pol, nu, a.mean(), flush=True)
        json.dump(dict(out=out, e_rate=E_RATE, rule={str(pi): rule(pi) for pi in PIS}), open(f'{OUT}/corr_synth.json', 'w'), indent=1)
    if 'nu_cv' in what:
        import experiments as ex
        res = {}
        for nu in (0.14, 0.25, 0.32, 0.5, 1.0):
            ex.BEL = [nu]
            r, _ = ex.sweep([(str(B), B, {}) for B in (0.0005, 0.002)], ['cadpp'])
            res[str(nu)] = {B: dict(aoii=r[B]['cadpp']['aoii'], ci=r[B]['cadpp']['aoii_ci'], energy=r[B]['cadpp']['energy'], runs=r[B]['cadpp']['runs']) for B in r}
            print('nu', nu, {B: res[str(nu)][B]['aoii'] for B in r}, flush=True)
        json.dump(res, open(f'{OUT}/nu_cv.json', 'w'), indent=1)
