#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_maxwell_connect.py
======================

Connect EoS branches of DIFFERENT nuclear composition (different proton/neutron
content, e.g. eos_8,8.dat ... eos_15,1.dat, eos_rmf.dat) into a single cold EoS by
a proper **Maxwell construction**:

    two branches coexist where   p_A = p_B  AND  mu_A = mu_B ,

and at a given chemical potential the *stable* branch is the one with the
**maximum pressure** (equivalently, at a given pressure, the one with the minimum
Gibbs free energy per baryon g = mu = (e+p)/n).

This replaces the previous "lowest energy density at fixed baryon density"
(energy-envelope) + tanh-crossover scheme, which does not enforce mu_A = mu_B and
introduces spurious pressure kinks at a first-order composition transition.

See EOS_maxwell_construction_notes.md for the physics.

Data conventions (current eosdata/):
    columns  = [ n_B [fm^-3] ,  p [MeV/fm^3] ,  e [MeV/fm^3] ]
    eos_(Z,N).dat : energy density EXCLUDES nucleon rest mass  (e/n ~ 0.3 MeV)
    eos_rmf.dat   : energy density INCLUDES rest mass          (e/n ~ 939.6 MeV)
The rest-mass offset is auto-detected and unified (see unify_rest_mass()).

Author: (rewrite of the composition-connection step)
"""

import os
import numpy as np

M_NUCLEON = 939.0          # MeV, rest mass added to bring branches to a common zero
REST_MASS_THRESHOLD = 100.0  # MeV/baryon; below this, a branch is assumed rest-mass-subtracted
MU_MAX_PHYSICAL = 2000.0     # MeV/baryon; reject rows above this (corrupted tails)


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_branch(path, m_nucleon=M_NUCLEON, label=None):
    """Load one EoS branch and return a clean, rest-mass-unified dict.

    Returns dict with keys: label, n, p, e, mu  (all 1-D, ascending in mu),
    where mu = (e+p)/n is the Gibbs free energy per baryon [MeV].
    """
    d = np.loadtxt(path)
    if d.shape[1] == 4:
        # composition tables eos_(N,Z).dat: [n_B, n_cell = n_B/A, p, e]
        # (the 2nd column is the Wigner-Seitz CELL number density, NOT pressure)
        n, p, e = d[:, 0].astype(float), d[:, 2].astype(float), d[:, 3].astype(float)
    else:
        # eos_rmf.dat: [n_B, p, e]
        n, p, e = d[:, 0].astype(float), d[:, 1].astype(float), d[:, 2].astype(float)

    # 1) drop non-finite rows (some files have NaNs near the high-density end)
    good = np.isfinite(n) & np.isfinite(p) & np.isfinite(e)
    n, p, e = n[good], p[good], e[good]

    # 2) physical sanity: positive density, non-negative pressure
    good = (n > 0) & (p >= 0) & (e > 0)
    n, p, e = n[good], p[good], e[good]

    # 3) unify rest-mass convention -> mu on the ~939 MeV scale
    e = _add_rest_mass_if_needed(n, e, m_nucleon)

    # 4) sort by density and enforce strict monotonicity in p (thermodynamic stab.)
    order = np.argsort(n)
    n, p, e = n[order], p[order], e[order]
    keep = np.concatenate(([True], np.diff(p) > 0))
    n, p, e = n[keep], p[keep], e[keep]

    mu = (e + p) / n
    # reject unphysical / corrupted rows (some high-density tails blow up to
    # mu ~ 1e4-1e5 MeV). Corrupted points always have HIGHER mu, so they simply
    # lose the min-mu selection; this cap just keeps interpolation well-behaved.
    good = mu < MU_MAX_PHYSICAL
    n, p, e, mu = n[good], p[good], e[good], mu[good]

    # 5) drop corrupted stretches: on a physical branch mu(p) increases smoothly
    #    (dmu/dp = 1/n > 0) and stays on the ~939-1000 MeV scale. Break the data
    #    at any decrease (beyond noise) or any discontinuous jump (> 5 MeV
    #    between adjacent rows), then keep the LONGEST clean segment. This
    #    removes both noisy leading rows and blown-up high-density tails.
    if mu.size > 2:
        dmu = np.diff(mu)
        brk = (dmu < -1e-7 * np.abs(mu[:-1])) | (dmu > 5.0)
        idx = np.concatenate(([0], np.where(brk)[0] + 1, [mu.size]))
        seg = max(zip(idx[:-1], idx[1:]), key=lambda ab: ab[1] - ab[0])
        n, p, e, mu = n[seg[0]:seg[1]], p[seg[0]:seg[1]], e[seg[0]:seg[1]], mu[seg[0]:seg[1]]

    return dict(label=label or os.path.basename(path), n=n, p=p, e=e, mu=mu)


def _add_rest_mass_if_needed(n, e, m_nucleon):
    """Return energy density on the rest-mass-included scale."""
    if np.median(e / n) < REST_MASS_THRESHOLD:
        return e + m_nucleon * n
    return e


# --------------------------------------------------------------------------- #
# p(mu) interpolation
# --------------------------------------------------------------------------- #
def _log_interp(x, xp, fp):
    """Monotone log-log linear interpolation of fp(xp) at x, no extrapolation
    (returns NaN outside [xp.min, xp.max]). xp assumed ascending & positive."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    inside = (x >= xp[0]) & (x <= xp[-1])
    lx, lxp, lfp = np.log(x[inside]), np.log(xp), np.log(fp)
    out[inside] = np.exp(np.interp(lx, lxp, lfp))
    return out


def branch_mu_of_p(branch, p_grid):
    """mu_i(p) on p_grid via log-log interpolation; NaN where branch undefined."""
    return _log_interp(p_grid, branch["p"], branch["mu"])


# --------------------------------------------------------------------------- #
# Exact pairwise Maxwell point: solve p_A = p_B AND mu_A = mu_B
# --------------------------------------------------------------------------- #
def pair_crossing(bA, bB, n_scan=4000):
    """Exact Maxwell point between two branches.

    Solves mu_A(p) - mu_B(p) = 0 by bisection in log p (both equilibrium
    conditions hold at the root: same p by construction, same mu by the root).
    Returns a transition dict, or None if the branches never cross in their
    common pressure window.
    """
    p_lo = max(bA["p"][0], bB["p"][0])
    p_hi = min(bA["p"][-1], bB["p"][-1])
    if p_hi <= p_lo:
        return None

    pg = np.logspace(np.log10(p_lo), np.log10(p_hi), n_scan)
    d = branch_mu_of_p(bA, pg) - branch_mu_of_p(bB, pg)
    ok = np.isfinite(d)
    pg, d = pg[ok], d[ok]
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if s.size == 0:
        return None

    # bisection on the FIRST sign change (physical transition = lowest pressure)
    a, b = np.log(pg[s[0]]), np.log(pg[s[0] + 1])
    fa = branch_mu_of_p(bA, np.array([np.exp(a)]))[0] - branch_mu_of_p(bB, np.array([np.exp(a)]))[0]
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = branch_mu_of_p(bA, np.array([np.exp(m)]))[0] - branch_mu_of_p(bB, np.array([np.exp(m)]))[0]
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    p_t = float(np.exp(0.5 * (a + b)))
    mu_t = float(branch_mu_of_p(bA, np.array([p_t]))[0])

    return dict(
        p_t=p_t, mu_t=mu_t,
        branch_lo=bA["label"], branch_hi=bB["label"],
        n_lo=float(_log_interp(np.array([p_t]), bA["p"], bA["n"])[0]),
        n_hi=float(_log_interp(np.array([p_t]), bB["p"], bB["n"])[0]),
        e_lo=float(_log_interp(np.array([p_t]), bA["p"], bA["e"])[0]),
        e_hi=float(_log_interp(np.array([p_t]), bB["p"], bB["e"])[0]),
    )


def maxwell_connect_sequence(branches, n_p=6000):
    """Connect branches in the GIVEN order, joining every consecutive pair at
    its exact Maxwell point (p_A = p_B, mu_A = mu_B).

    Use this when the composition sequence is prescribed (e.g. 8,8 -> 9,7 ->
    ... -> rmf). Pairs that never cross are reported in 'no_crossing' and the
    earlier branch is simply followed until the next available transition.
    Returns the same dict structure as maxwell_connect().
    """
    transitions, no_crossing = [], []
    for bA, bB in zip(branches[:-1], branches[1:]):
        t = pair_crossing(bA, bB)
        if t is None:
            no_crossing.append((bA["label"], bB["label"]))
        else:
            transitions.append(t)

    # keep only transitions at increasing pressure (thermodynamic consistency)
    transitions.sort(key=lambda t: t["p_t"])
    kept, p_prev = [], -np.inf
    for t in transitions:
        if t["p_t"] > p_prev:
            kept.append(t); p_prev = t["p_t"]
    transitions = kept

    # assemble the piecewise curve on a global pressure grid
    p_lo = branches[0]["p"][0]
    p_hi = branches[-1]["p"][-1]
    p_grid = np.logspace(np.log10(p_lo), np.log10(p_hi), n_p)

    labels = [b["label"] for b in branches]
    bounds = [t["p_t"] for t in transitions]           # segment boundaries
    seq = [labels.index(transitions[0]["branch_lo"])] if transitions else [0]
    for t in transitions:
        seq.append(labels.index(t["branch_hi"]))

    winner = np.empty(p_grid.size, dtype=int)
    for k, p in enumerate(p_grid):
        i = np.searchsorted(bounds, p)
        winner[k] = seq[min(i, len(seq) - 1)]

    n_env = np.empty_like(p_grid); e_env = np.empty_like(p_grid); mu_env = np.empty_like(p_grid)
    for k, (p, w) in enumerate(zip(p_grid, winner)):
        b = branches[w]
        n_env[k] = _log_interp(np.array([p]), b["p"], b["n"])[0]
        e_env[k] = _log_interp(np.array([p]), b["p"], b["e"])[0]
        mu_env[k] = branch_mu_of_p(b, np.array([p]))[0]

    good = np.isfinite(n_env) & np.isfinite(e_env)
    return dict(p=p_grid[good], mu=mu_env[good], n=n_env[good], e=e_env[good],
                winner=winner[good], transitions=transitions,
                no_crossing=no_crossing)


# --------------------------------------------------------------------------- #
# Maxwell construction, cast at fixed pressure
# --------------------------------------------------------------------------- #
# At fixed p the stable branch minimises the Gibbs energy per baryon mu(p).
# This is identical physics to "max pressure at fixed mu" but far better
# conditioned for this dataset: pressure spans ~12 decades cleanly while all mu
# sit in a narrow ~939-960 MeV window. A composition transition occurs where two
# branches' mu(p) curves cross: there p_A = p_B (same grid point) and
# mu_A = mu_B (the crossing), i.e. exactly the Maxwell conditions, with a density
# jump n_A(p_t) -> n_B(p_t).
def maxwell_connect(branches, n_p=6000, p_min=None, p_max=None):
    """Build the connected cold EoS as the lower envelope min_i mu_i(p).

    Returns dict with:
        p, mu, n, e   : stable-branch curve on the pressure grid
        winner        : index of the stable branch at each pressure
        transitions   : list of dicts (mu_t, p_t, branch_lo/hi, n_lo/hi, e_lo/hi)
    """
    p_lo = p_min if p_min else max(b["p"][0] for b in branches)
    p_hi = p_max if p_max else min(b["p"][-1] for b in branches)
    p_grid = np.logspace(np.log10(p_lo), np.log10(p_hi), n_p)

    # mu_i(p) for every branch; NaN -> +inf so it never wins where undefined
    MU = np.vstack([branch_mu_of_p(b, p_grid) for b in branches])
    MU_masked = np.where(np.isfinite(MU), MU, +np.inf)

    winner = np.argmin(MU_masked, axis=0)          # stable branch = MIN Gibbs/baryon
    mu_env = MU_masked[winner, np.arange(p_grid.size)]

    # recover n, e on the winning branch at each pressure
    n_env = np.empty_like(p_grid)
    e_env = np.empty_like(p_grid)
    for k, (p, w) in enumerate(zip(p_grid, winner)):
        b = branches[w]
        n_env[k] = _log_interp(np.array([p]), b["p"], b["n"])[0]
        e_env[k] = _log_interp(np.array([p]), b["p"], b["e"])[0]

    # locate transitions (winner changes) and solve each one EXACTLY:
    # pair_crossing() finds the root of mu_A(p) - mu_B(p), i.e. the point where
    # p_A = p_B and mu_A = mu_B hold simultaneously (Maxwell conditions).
    transitions = []
    for s in np.where(np.diff(winner) != 0)[0]:
        lo, hi = branches[winner[s]], branches[winner[s + 1]]
        t = pair_crossing(lo, hi)
        if t is None:
            # No exact root in the bracket: never fabricate a junction (a wrong
            # transition silently corrupts the TOV table). Report and skip.
            print(f"WARNING: winner switches {lo['label']} -> {hi['label']} near "
                  f"p~{p_grid[s]:.3e} but mu_A(p)=mu_B(p) has no root there; "
                  f"transition NOT emitted - inspect the branches.")
            continue
        # physical sanity: with increasing pressure the stable phase must be
        # denser and more energetic (first-order transition, one conserved charge)
        if not (t["n_hi"] > t["n_lo"] and t["e_hi"] > t["e_lo"]):
            print(f"WARNING: rejected transition {t['branch_lo']} -> {t['branch_hi']} "
                  f"at p_t={t['p_t']:.3e}: non-physical jump "
                  f"(dn={t['n_hi']-t['n_lo']:+.3e}, de={t['e_hi']-t['e_lo']:+.3e}).")
            continue
        transitions.append(t)

    return dict(p=p_grid, mu=mu_env, n=n_env, e=e_env,
                winner=winner, transitions=transitions)


# --------------------------------------------------------------------------- #
# Assemble a TOV-ready (n, p, e) table with explicit density jumps
# --------------------------------------------------------------------------- #
def to_tov_table(result, insert_jumps=True):
    """Return array [n, p, e] sorted by pressure, with each first-order
    transition represented by two rows at the SAME pressure (Maxwell jump)."""
    n, p, e = result["n"].copy(), result["p"].copy(), result["e"].copy()
    rows = list(zip(n, p, e))

    if insert_jumps:
        for t in result["transitions"]:
            # two coexistence points at equal p_t: (n_lo,e_lo) then (n_hi,e_hi)
            rows.append((t["n_lo"], t["p_t"], t["e_lo"]))
            rows.append((t["n_hi"], t["p_t"], t["e_hi"]))

    arr = np.array(rows, float)
    arr = arr[np.argsort(arr[:, 1], kind="stable")]     # ascending pressure
    return arr


# --------------------------------------------------------------------------- #
# Demo / CLI
# --------------------------------------------------------------------------- #
def _demo(eosdir="eosdata", out="eosdata/eos_maxwell_connected.dat"):
    fam = ["eos_8,8.dat", "eos_9,7.dat", "eos_10,6.dat", "eos_11,5.dat",
           "eos_12,4.dat", "eos_13,3.dat", "eos_14,2.dat", "eos_15,1.dat",
           "eos_rmf.dat"]
    branches = []
    for f in fam:
        path = os.path.join(eosdir, f)
        if not os.path.exists(path):
            print("  (skip, missing)", f); continue
        b = load_branch(path, label=f.replace("eos_", "").replace(".dat", ""))
        if b["n"].size < 10:
            print("  (skip, too few clean rows)", f); continue
        branches.append(b)
        print(f"  loaded {b['label']:8s}  rows={b['n'].size:6d}  "
              f"mu:[{b['mu'][0]:.2f},{b['mu'][-1]:.2f}]  p:[{b['p'][0]:.2e},{b['p'][-1]:.2e}]")

    res = maxwell_connect(branches)
    print("\nStable composition sequence (low -> high mu):")
    seq, cur = [], None
    for w in res["winner"]:
        if w != cur:
            seq.append(branches[w]["label"]); cur = w
    print("  " + "  ->  ".join(seq))

    print("\nFirst-order composition transitions:")
    for t in res["transitions"]:
        print(f"  {t['branch_lo']:>6s} -> {t['branch_hi']:<6s} "
              f"at mu={t['mu_t']:.3f} MeV, p={t['p_t']:.4e} MeV/fm^3, "
              f"n: {t['n_lo']:.3e} -> {t['n_hi']:.3e} fm^-3 "
              f"(Delta n/n = {(t['n_hi']-t['n_lo'])/t['n_lo']:+.1%})")

    table = to_tov_table(res)
    np.savetxt(out, table, fmt="%.18e")
    print(f"\nwrote {out}  ({table.shape[0]} rows, columns: n p e)")
    return res, table


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "eosdata"
    _demo(eosdir=d)
