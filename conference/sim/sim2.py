"""Trace-driven simulator: real Sen1Floods11 inference traces + UAV air-to-ground link.
Policies: CA-DPP (proposed) and its ablations, and baselines adapted from
Gan et al. 2025, Pehlivanoglu et al. 2026, Basnayaka et al. 2025, Zakeri et al. 2023."""
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
TR = dict(np.load(os.path.join(HERE, 'traces_sen1floods11.npz')))

P = dict(Dmax=300, H=100.0, R=1300.0, cx=1500.0, period=400, fc=2.4e9, W=1e6, Ptx_dBm=20.0, NF=7.0,
         aL=9.61, bL=0.16, etaL=1.0, etaN=20.0, L1=256.0, L2=4e5, tmax=0.4, Pr=1.0, Eo=0.02,
         p_block=0.02, p_unblock=0.1, credit=0.5, onboard='on8')

def link_tables(P):
    k = np.arange(P['period']); ang = 2*np.pi*k/P['period']
    x = P['cx'] + P['R']*np.cos(ang); y = P['R']*np.sin(ang)
    dh = np.hypot(x, y); d3 = np.hypot(dh, P['H']); phi = np.degrees(np.arctan2(P['H'], dh))
    pl = 1/(1+P['aL']*np.exp(-P['bL']*(phi-P['aL'])))
    PL = 20*np.log10(d3) + 20*np.log10(P['fc']) - 147.55 + pl*P['etaL'] + (1-pl)*P['etaN']
    snr = 10**((P['Ptx_dBm'] - PL - (-174 + 10*np.log10(P['W']) + P['NF']))/10)
    out = dict(dist=dh, snr_db=10*np.log10(snr))
    for a, L in ((1, P['L1']), (2, P['L2'])):
        out[f'q{a}'] = np.exp(-(2**(L/(P['W']*P['tmax'])) - 1)/snr)
        out[f'E{a}'] = P['Pr']*np.minimum(L/(P['W']*np.log2(1+snr)), P['tmax']) + P['Eo']
    return out

def model_params(P):
    """Quantities the controller may use, all estimated on train/validation data only."""
    pv = TR[P['onboard']+'_va']; Xv = TR['X_va']; gv = TR['gr_va']
    cv = np.maximum(pv, 1-pv); wrong_g = (gv >= .5) != Xv
    edges = np.array([.5, .6, .7, .8, .9, .95, .99, 1.0001])
    b = np.clip(np.digitize(cv, edges)-1, 0, len(edges)-2)
    eps_e = np.array([wrong_g[b == i].mean() if (b == i).sum() > 20 else wrong_g.mean() for i in range(len(edges)-1)])
    acc_on = ((pv >= .5) == Xv).mean()
    return dict(p=np.array([float(TR['p01']), float(TR['p10'])]), edges=edges, eps_e=eps_e,
                eps_e_mean=wrong_g.mean(), acc_on=acc_on)

def make_trace(seed, reps=4):
    """Test chips visited in random order, `reps` passes (each pass a new random order)."""
    rng = np.random.default_rng(seed); ch = TR['chip']
    idx = np.concatenate([np.where(ch == c)[0] for _ in range(reps) for c in rng.permutation(np.unique(ch))])
    return idx, rng

def run(policy, budget, par=None, seed=1, P=P, mixlog=False):
    LT = link_tables(P); MP = model_params(P); per = P['period']; Dm = P['Dmax']
    pflip = MP['p']; idx, rng = make_trace(seed); T = len(idx)
    X_tr = TR['X'][idx]; pon = TR[P['onboard']+'_te'][idx]; pgr = TR['gr_te'][idx]
    ul = rng.random(T); ub = rng.random(T)
    blocked = np.zeros(T, bool); st = False
    for t in range(T):
        st = (ub[t] >= P['p_unblock']) if st else (ub[t] < P['p_block']); blocked[t] = st
    delta = np.arange(Dm+1); up = np.minimum(delta+1, Dm)
    b = np.zeros(Dm+1); b[0] = 1.0
    Xh = 0; D = 0; Z = 0.0; since = 0; run_len = 0
    sumD = sumE = 0.0; wrong = 0; cnt = np.zeros(3); mix = np.zeros((per, 3)); Ein = 0.0
    for t in range(T):
        X = X_tr[t]; k = t % per
        q = (0.0, LT['q1'][k], LT['q2'][k]) if not blocked[t] else (0.0, 0.0, 0.0)
        E = (0.0, LT['E1'][k], LT['E2'][k])
        pr = pon[t]; lab = int(pr >= .5); c = max(pr, 1-pr)
        d = int(lab != Xh)
        run_len = run_len + 1 if d else 0
        # ---------- belief correction with the on-board label ----------
        cc = c if policy != 'cadpp_blind' else MP['acc_on']
        Lp = cc if d else 1-cc
        b[0] *= (1-Lp); b[1:] *= Lp; b /= b.sum()
        xw = 1-Xh                        # value of X if receiver is wrong
        pr_r, pr_w = pflip[Xh], pflip[xw]  # flip prob. from the receiver's state / the other state
        epsE = MP['eps_e'][np.clip(np.digitize(c, MP['edges'])-1, 0, 6)] if policy != 'cadpp_blind' else MP['eps_e_mean']
        # one-step expected AoII (general asymmetric form, Lemma 1)
        w0 = (delta+1.0)
        J0 = b[0]*w0[0]*pr_r + np.dot(b[1:], w0[1:])*(1-pr_w)
        J1 = (b[0]*w0[0]*(1-pr_r) + np.dot(b[1:], w0[1:])*pr_w) if d else J0
        J2 = b[0]*w0[0]*((1-epsE)*pr_r + epsE*(1-pr_r)) + np.dot(b[1:], w0[1:])*((1-epsE)*pr_w + epsE*(1-pr_w))
        G = (0.0, J0-J1, J0-J2)
        # ---------- decision ----------
        a = 0
        if q[1] > 0:
            if policy.startswith('cadpp'):
                u = [0.0, par*q[1]*G[1] - Z*E[1], par*q[2]*G[2] - Z*E[2]]
                if policy == 'cadpp_label': u[2] = -np.inf
                a = int(np.argmax(u))
            elif policy == 'gan':        # pre-identification: report every frame, upload raw if target seen
                a = 2 if lab == 1 else 1
            elif policy == 'pehl':       # age-of-inference threshold, fixed success probabilities
                A, qs = par
                if since >= A: a = 2 if q[2] >= qs else 1
            elif policy == 'basn':       # always offload features for ground inference, periodic
                if since >= par: a = 2
            elif policy == 'oracle':     # genie: knows X(t), sends an error-free update on mismatch
                a = 1 if X != Xh else 0
            elif policy == 'zakeri':     # AoII threshold, treating the on-board label as the truth
                if run_len >= par: a = 1
            else: raise ValueError(policy)
            if not policy.startswith('cadpp') and a > 0 and Ein + E[a] > budget*(t+1) + P['credit']:
                a = 1 if (a == 2 and Ein + E[1] <= budget*(t+1) + P['credit'] and policy != 'basn') else 0
        cnt[a] += 1
        if mixlog and not blocked[t]: mix[k, a] += 1
        Ein += E[a]; sumE += E[a]; Z = max(Z + E[a] - budget, 0.0)
        ok = a > 0 and ul[t] < q[a]
        newXh = Xh
        if ok:
            if policy == 'oracle': newXh = X
            elif a == 1: newXh = lab
            else:
                le = int(pgr[t] >= .5); de = int(le != Xh)
                Lp = (1-epsE) if de else epsE
                b[0] *= (1-Lp); b[1:] *= Lp; b /= b.sum()
                newXh = le
            since = 0
        else: since += 1
        # ---------- belief prediction ----------
        changed = newXh != Xh; nb = np.zeros_like(b)
        pm = pflip[newXh]; px = pflip[1-newXh]   # flip prob when X(t)=new estimate / otherwise
        match0 = not changed                     # does delta=0 mean X(t)=new estimate?
        if match0:
            nb[0] += b[0]*(1-pm); nb[1] += b[0]*pm
            nb[0] += b[1:].sum()*px; np.add.at(nb, up[1:], b[1:]*(1-px))
        else:
            nb[1] += b[0]*(1-px); nb[0] += b[0]*px
            nb[0] += b[1:].sum()*(1-pm); np.add.at(nb, up[1:], b[1:]*pm)
        b = nb/nb.sum()
        Xh = newXh
        if t+1 < T:
            Xn = X_tr[t+1]
            D = (D+1) if Xn != Xh else 0
        sumD += D; wrong += (X_tr[min(t+1, T-1)] != Xh)
        if lab == Xh: run_len = 0        # the delivered label now matches
    return dict(policy=policy, par=par, budget=budget, aoii=sumD/T, wrong=wrong/T, energy=sumE/T,
                frac=(cnt/T).tolist(), Zend=Z, mix=mix.tolist() if mixlog else None, T=T)
