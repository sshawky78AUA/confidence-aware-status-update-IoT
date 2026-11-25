"""Optimality gap of CA-DPP in a model that satisfies Assumption 1 (conditionally independent classifier outputs).
Usage: python3 optgap.py OUTDIR

Synthetic model: X(t) is the two-state Markov chain estimated from the data; in each slot the classifier
outputs (on-board probability, ground probability) are drawn jointly from the pooled validation+test slots
with the same value of X, so errors are independent over time given X. Orbit, link, blockage and energy are
those of the scenario.

Lower bound: the full-information relaxation. A controller that also observes X(t), and hence the AoII and
whether the on-board label is right, can imitate any policy of (P), so its optimal constrained cost is a lower
bound on the optimum of (P). We solve its Lagrangian relaxation exactly by dynamic programming over
(X, AoII, orbit position, blockage) and use weak duality, LB(B) = max_lambda g(lambda) - lambda*B, where g is the
optimal average Lagrangian cost. The AoII is capped at DCAP inside the bound, which can only lower it.
CA-DPP and the AoII-threshold policy are simulated in the same synthetic model, with V or k tuned per budget."""
import numpy as np, json, os, sys
import sim
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results'; os.makedirs(OUT, exist_ok=True)
P = dict(sim.P); TR = sim.TR
DCAP = 150
EDGES = np.array([.5, .6, .7, .8, .9, .95, .99, 1.0001]); NB = len(EDGES)-1

X_all = TR['X_all'].astype(int); pon = TR['on8_all']; pgr = TR['gr_all']
lab = (pon >= .5).astype(int); c = np.maximum(pon, 1-pon); cb = np.clip(np.digitize(c, EDGES)-1, 0, NB-1)
corr = (lab == X_all).astype(int); gcor = ((pgr >= .5).astype(int) == X_all).astype(int)
p01, p10 = float(TR['p01']), float(TR['p10']); pflip = np.array([p01, p10])

# observation model per state x: o = (corr, bin); P(o | x) and P(ground right | x, o)
Po = np.zeros((2, 2, NB)); Gc = np.zeros((2, 2, NB))
for x in (0, 1):
    m = X_all == x
    for r in (0, 1):
        for b in range(NB):
            mm = m & (corr == r) & (cb == b); Po[x, r, b] = mm.sum()/m.sum()
            Gc[x, r, b] = gcor[mm].mean() if mm.sum() > 0 else gcor[m].mean()

def lagrangian_dp(lam, LT, orbits=30):
    """Backward DP over time; returns average Lagrangian cost per slot and the stationary policy tables."""
    per = P['period']; D = np.arange(DCAP+1); pb, pu = P['p_block'], P['p_unblock']
    V = np.zeros((2, 2, DCAP+1))              # V[b, X, Delta] at orbit position k (start of slot, after blockage draw)
    hist = []
    for n in range(orbits*per):
        k = (per - 1 - n) % per; kn = (k+1) % per
        # expected continuation over next blockage state, as a function of (X', Delta') at position kn
        Vb = np.stack([(1-pb)*V[0] + pb*V[1], pu*V[0] + (1-pu)*V[1]])   # Vb[b](X', D')
        Dn = np.minimum(D+1, DCAP)
        U = np.zeros((2, 2, 2, DCAP+1))       # U[b, r, X, Delta]: r = 1 if the estimate after the action equals X
        for x in (0, 1):
            f = pflip[x]
            # r=1: wrong next slot only if X flips (estimate = x, X' = 1-x)
            U[:, 1, x, :] = f*(Dn[None, :] + Vb[:, 1-x, :][:, Dn]) + (1-f)*(0 + Vb[:, x, 0][:, None])
            # r=0: estimate = 1-x; wrong next slot if X stays
            U[:, 0, x, :] = (1-f)*(Dn[None, :] + Vb[:, x, :][:, Dn]) + f*(0 + Vb[:, 1-x, 0][:, None])
        r0 = (D == 0).astype(int)
        U0 = np.where(r0[None, None, :] == 1, U[:, 1], U[:, 0])          # [b, X, D] silence
        newV = np.zeros_like(V)
        q1, q2, E1, E2 = LT['q1'][k], LT['q2'][k], LT['E1'][k], LT['E2'][k]
        for b in (0, 1):
            qq1, qq2 = (q1, q2) if b == 0 else (0.0, 0.0)
            for x in (0, 1):
                acc = np.zeros(DCAP+1)
                for r in (0, 1):
                    for bi in range(NB):
                        Q0 = U0[b, x]
                        Q1 = lam*E1 + qq1*U[b, r, x] + (1-qq1)*U0[b, x]
                        g = Gc[x, r, bi]
                        Q2 = lam*E2 + qq2*(g*U[b, 1, x] + (1-g)*U[b, 0, x]) + (1-qq2)*U0[b, x]
                        Q = np.minimum(Q0, np.minimum(Q1, Q2)) if b == 0 else Q0
                        acc += Po[x, r, bi]*Q
                newV[b, x] = acc
        V = newV
        if k == 0: hist.append(V[0, 0, 0])
    g = (hist[-1] - hist[-11])/(10*per)
    return g

def lower_bound(B, LT, lams):
    vals = [(lagrangian_dp(l, LT) - l*B, l) for l in lams]
    return max(vals)

# ---------------- synthetic traces and policy simulation ----------------
def synth(seed, T):
    rng = np.random.default_rng(seed); X = np.zeros(T, int); X[0] = rng.random() < p01/(p01+p10)
    u = rng.random(T)
    for t in range(1, T): X[t] = (1-X[t-1]) if u[t] < pflip[X[t-1]] else X[t-1]
    idx = np.empty(T, int)
    for x in (0, 1):
        pool = np.where(X_all == x)[0]; m = X == x; idx[m] = rng.choice(pool, m.sum())
    return X, pon[idx], pgr[idx]

def run_synth(pol, B, par, seed, T=100000, energy="mean"):
    X, po, pg = synth(seed, T)
    TR['X_syn'], TR['on8_syn'], TR['gr_syn'] = X, po, pg
    old = sim.make_trace
    sim.make_trace = lambda s, reps=4, split='te': (np.arange(T), np.random.default_rng(seed+7))
    try:
        PP = dict(P); PP['energy'] = energy
        r = sim.run(pol, B, par, seed, P=PP, split='syn')
    finally:
        sim.make_trace = old
    return r

def job(a):
    pol, B, par, s = a; r = run_synth(pol, B, par, s); return (pol, B, json.dumps(par), s), (r['aoii'], r['energy'])

if __name__ == '__main__':
    from multiprocessing import Pool
    LT = sim.link_tables(P)
    budgets = [0.00025, 0.0005, 0.001, 0.002]
    # CA-DPP with nu=1 (independent errors: no tempering needed) and the prior correction; AoII threshold k
    Vg = {0.00025: [0.0003, 0.001, 0.003, 0.01], 0.0005: [0.003, 0.01, 0.03, 0.1], 0.001: [0.01, 0.03, 0.1, 0.3], 0.002: [0.1, 0.3, 1, 3]}
    jobs = [('cadpp', B, (v, 1.0, 0), s) for B in budgets for v in Vg[B] for s in (11, 12)]
    jobs += [('zakeri', B, k, s) for B in budgets for k in (1, 2, 3, 5) for s in (11, 12)]
    with Pool(int(os.environ.get('NPROC', 2))) as p: TU = dict(p.map(job, jobs))
    def pick(pol, B, grid):
        best = None
        for par in grid:
            rr = [TU[(pol, B, json.dumps(par), s)] for s in (11, 12)]; a = np.mean([r[0] for r in rr]); e = np.mean([r[1] for r in rr])
            key = (0 if e <= 1.05*B else 1, a if e <= 1.05*B else e)
            if best is None or key < best[0]: best = (key, par)
        return best[1]
    chosen = {B: {'cadpp': pick('cadpp', B, [(v, 1.0, 0) for v in Vg[B]]), 'zakeri': pick('zakeri', B, [1, 2, 3, 5])} for B in budgets}
    ev = [(pol, B, chosen[B][pol], s) for B in budgets for pol in ('cadpp', 'zakeri') for s in (1, 2, 3, 4, 5, 6)]
    with Pool(int(os.environ.get('NPROC', 2))) as p: EV = dict(p.map(job, ev))
    lams = np.concatenate([[0], np.logspace(0, 4.5, 30)])     # price per joule (AoII-slots per J)
    G = {float(l): lagrangian_dp(l, LT) for l in lams}
    out = {}
    for B in budgets:
        res = {}
        for pol in ('cadpp', 'zakeri'):
            rr = [EV[(pol, B, json.dumps(chosen[B][pol]), s)] for s in (1, 2, 3, 4, 5, 6)]
            a = np.array([r[0] for r in rr]); e = np.array([r[1] for r in rr])
            Eu = e.mean()                                    # bound evaluated at the energy actually spent
            lb = max(G[l] - l*Eu for l in G)
            res[pol] = dict(par=chosen[B][pol], aoii=float(a.mean()), aoii_sd=float(a.std(ddof=1)), energy=float(Eu), lb=float(lb), gap=float((a.mean()-lb)/lb))
        res['lb_at_B'] = float(max(G[l] - l*B for l in G))
        out[str(B)] = res; print(B, res, flush=True)
    json.dump(dict(out=out, G=G, Po=Po.tolist(), Gc=Gc.tolist()), open(f'{OUT}/optgap.json', 'w'), indent=1)
