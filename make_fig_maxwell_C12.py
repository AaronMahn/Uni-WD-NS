#!/usr/bin/env python3
"""Regenerate Fig. 1 (Maxwell construction for 12C) as TWO separate panels
   (fig_maxwell_C12a.png, fig_maxwell_C12b.png) for a subfigure layout.
   Improvements over the old combined fig_maxwell_C12.png:
     - p_t^(2) label moved next to its star (no longer floating over (9,3) label)
     - (9,3) branch label placed on the green curve, away from p_t^(2)
     - eps_- / eps_+ labels offset from the dotted guides (no clipping)
     - minimum-Gibbs envelope drawn as a visible thick gray halo
     - no in-panel (a)/(b) tags: subcaption provides them below each panel
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos_maxwell_connect import load_branch, pair_crossing, _log_interp

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'eosdata')                      # WDprog/eosdata
OUT  = os.environ.get('WDNS_FIG_DIR', os.path.join(HERE, 'figures'))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset":   "stix",
    "font.size":          14,
    "axes.labelsize":     16,
    "axes.titlesize":     16,
    "axes.linewidth":     1.1,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.minor.size":   3,
    "ytick.minor.size":   3,
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "legend.frameon":     False,
    "lines.linewidth":    2.0,
    "figure.dpi":         110,
    "savefig.dpi":        500,
    "savefig.bbox":       "tight",
})

# ----- data ---------------------------------------------------------------
b66  = load_branch(f'{BASE}/eos_c/eos_6,6=.dat', label='(6,6)')
b75  = load_branch(f'{BASE}/eos_c/eos_7,5.dat',  label='(7,5)')
b84  = load_branch(f'{BASE}/eos_c/eos_8,4.dat',  label='(8,4)')
b93  = load_branch(f'{BASE}/eos_c/eos_9,3.dat',  label='(9,3)')
b102 = load_branch(f'{BASE}/eos_c/eos_10,2.dat', label='(10,2)')
b111 = load_branch(f'{BASE}/eos_c/eos_11,1.dat', label='(11,1)')
brmf = load_branch(f'{BASE}/eos_rmf.dat',        label='uniform')

t1 = pair_crossing(b66, b75)
t2 = pair_crossing(b75, b84)
t3 = pair_crossing(b84, b93)
t4 = pair_crossing(b93, brmf)
for k, t in enumerate((t1, t2, t3, t4), 1):
    print(f"p_t({k}) = {t['p_t']:.3e} MeV/fm^3, mu_t = {t['mu_t']:.2f} MeV, "
          f"eps: {t['e_lo']:.4f} -> {t['e_hi']:.4f}")

C66, C75, C84, C93 = '#d97fa8', '#e2a33c', '#2b6fb3', '#2f9e6e'
C102, C111 = '0.55', '0.75'

MU_LO, MU_HI = 933.45, 941.5
P_LO,  P_HI  = 2.0e-11, 1.0e-3

def crop(b):
    m = (b['mu'] >= MU_LO - 0.2) & (b['mu'] <= MU_HI + 0.2) & \
        (b['p'] >= P_LO * 0.5) & (b['p'] <= P_HI * 2.5)
    return b['mu'][m], b['p'][m]

# ----- panel (a): p - mu_B -----------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.5))

# minimum-Gibbs (max-p at fixed mu) envelope: thick gray halo UNDER the curves
pieces = [(b66, None, t1), (b75, t1, t2), (b84, t2, t3), (b93, t3, t4), (brmf, t4, None)]
mu_env, p_env = [], []
for b, tlo, thi in pieces:
    lo = tlo['mu_t'] if tlo else -np.inf
    hi = thi['mu_t'] if thi else  np.inf
    m = (b['mu'] >= lo) & (b['mu'] <= hi)
    mu_env.append(b['mu'][m]); p_env.append(b['p'][m])
mu_env = np.concatenate(mu_env); p_env = np.concatenate(p_env)
ax.plot(mu_env, p_env, color='0.72', lw=7.0, alpha=0.95, zorder=1,
        solid_capstyle='round', label=None)

for b, c, ls, lw, z in ((b102, C102, '--', 1.8, 2), (b111, C111, ':', 1.8, 2),
                        (b66, C66, '-', 2.0, 3), (b75, C75, '-', 2.0, 3),
                        (b84, C84, '-', 2.0, 3), (b93, C93, '-', 2.0, 3),
                        (brmf, 'k', '-', 2.2, 3)):
    mu, p = crop(b)
    ax.plot(mu, p, color=c, ls=ls, lw=lw, zorder=z)

# stars + p_t labels, each label tied to ITS star
stars = ((t1, r'$p_t^{(1)}$', ( 0.06, 0.52), 'left',  'top'),
         (t2, r'$p_t^{(2)}$', (-0.62, 1.0 ), 'right', 'center'),
         (t3, r'$p_t^{(3)}$', (-0.55, 1.75), 'right', 'center'),
         (t4, r'$p_t^{(4)}$', ( 0.16, 0.50), 'left',  'top'))
for t, lab, (dx, fy), ha, va in stars:
    ax.plot(t['mu_t'], t['p_t'], marker='*', ms=17, color='k', zorder=6, ls='none')
    ax.text(t['mu_t'] + dx, t['p_t'] * fy, lab, ha=ha, va=va, fontsize=15, zorder=6)

def mu_at(b, p):
    return float(np.interp(np.log(p), np.log(b['p']), b['mu']))

# branch labels placed ON their own curves
ax.text(933.62, 4.5e-10, '(6,6)', color=C66, fontsize=15, ha='left')
ax.text(mu_at(b75, 2.0e-7) - 0.62, 2.0e-7, '(7,5)', color=C75, fontsize=15, ha='left')
ax.text(mu_at(b84, 3.0e-7) + 0.17, 3.0e-7, '(8,4)', color=C84, fontsize=15, ha='left')
ax.text(mu_at(b93, 1.0e-6) + 0.15, 1.0e-6, '(9,3)', color=C93, fontsize=15, ha='left')
ax.text(mu_at(b102, 1.0e-9) - 0.95, 1.0e-9, '(10,2)', color=C102, fontsize=15, ha='left')
ax.text(mu_at(b111, 3.0e-10) - 1.05, 3.0e-10, '(11,1)', color=C111, fontsize=15, ha='left')
ax.text(mu_at(brmf, 2.0e-8) - 1.15, 2.0e-8, 'uniform', color='k', fontsize=15, ha='left')

ax.set_yscale('log')
ax.set_xlim(MU_LO, MU_HI)
ax.set_ylim(P_LO, P_HI)
ax.set_xlabel(r'$\mu_B$ (MeV)')
ax.set_ylabel(r'$p$ (MeV fm$^{-3}$)')
fig.savefig(f'{OUT}/fig_maxwell_C12a.png')
plt.close(fig)

# ----- panel (b): p - eps zoom on (8,4)->(9,3) ---------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.5))
E_LO, E_HI = 0.045, 0.100
PB_LO, PB_HI = 0.4, 1.6          # units of 1e-4 MeV/fm^3
S = 1.0e4

for b, c in ((b84, C84), (b93, C93)):
    m = (b['e'] >= E_LO * 0.8) & (b['e'] <= E_HI * 1.2)
    ax.plot(b['e'][m], b['p'][m] * S, color=c, lw=2.4)

pt, em, ep = t3['p_t'] * S, t3['e_lo'], t3['e_hi']
# dotted guides from axis bottom to the plateau
ax.vlines([em, ep], PB_LO, pt, colors='0.6', ls=':', lw=1.4, zorder=2)
# constant-pressure Maxwell segment
ax.plot([em, ep], [pt, pt], color='k', lw=3.2, zorder=4,
        solid_capstyle='round')
ax.plot([em, ep], [pt, pt], marker='o', ms=9, mfc='white', mec='k', mew=1.8,
        ls='none', zorder=5)
ax.text(0.5 * (em + ep), pt + 0.035, r'$p_t$', ha='center', va='bottom', fontsize=16)
# eps labels OFFSET from the dotted lines so the subscripts are not clipped
ax.text(em - 0.0022, 0.50, '$\\varepsilon$₋', ha='right', va='center', fontsize=17)
ax.text(ep + 0.0022, 0.50, '$\\varepsilon$₊', ha='left',  va='center', fontsize=17)

ax.text(0.052, 1.32, '(8,4)', color=C84, fontsize=15)
ax.text(0.0930, 0.97, '(9,3)', color=C93, fontsize=15, ha='left', va='top')

ax.set_xlim(E_LO, E_HI)
ax.set_ylim(PB_LO, PB_HI)
ax.set_xlabel(r'$\varepsilon$ (MeV fm$^{-3}$)')
ax.set_ylabel(r'$p$ ($10^{-4}$ MeV fm$^{-3}$)')
fig.savefig(f'{OUT}/fig_maxwell_C12b.png')
plt.close(fig)
print('done')
