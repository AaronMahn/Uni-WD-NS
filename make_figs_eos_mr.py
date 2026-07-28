#!/usr/bin/env python3
"""Regenerate Figs. 2--5 (EOS and M-R panels)
   as separate per-panel PNGs for subcaption layout.
   - EOS panels from eosdata/eos_maxwell_pmu_*.dat (Maxwell jumps drawn explicitly)
   - M-R curves from dense TOV output (mr_dense_*.dat), smooth NS branches
   - Historical nonrotating HWW comparison from Hartle & Thorne (1968), Table 3
   - NS observations: J0030 (Kini 2026), J0740 (Salmi 2024), J0437 (Miller 2026),
     GW170817 90% common-EOS region, HESS J1731-347, J0952-0607 band 2.35+-0.11
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
EOSD = os.path.join(HERE, 'eosdata')
OBSD = os.path.join(HERE, 'input')
MRD  = os.path.join(HERE, 'results')
OUT  = os.environ.get('WDNS_FIG_DIR', os.path.join(HERE, 'figures'))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 14,
    "axes.labelsize": 16, "axes.linewidth": 1.1,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "legend.frameon": False, "legend.fontsize": 13,
    "lines.linewidth": 2.0, "figure.dpi": 110,
    "savefig.dpi": 500, "savefig.bbox": "tight",
})
CHE, CC, CO = '#e2a33c', '#2b6fb3', '#2f9e6e'
CTM = '#c1440e'

def load_eos(name):
    d = np.loadtxt(f'{EOSD}/{name}')
    n,p,e = d[:,0], d[:,1], d[:,2]
    g = np.isfinite(n)&np.isfinite(p)&np.isfinite(e)&(n>0)&(p>0)&(e>0)
    return n[g],p[g],e[g]

def jumps(n,p,e, thr=1.04):
    J=[]
    for i in range(1,len(p)):
        if p[i] <= p[i-1]*(1+1e-9) and n[i] > n[i-1]*thr:
            J.append((e[i-1], e[i], p[i-1]))
    return J

# ---------------- Fig 2: EOS panels ----------------
He = load_eos('eos_maxwell_pmu_He.dat')
C  = load_eos('eos_maxwell_pmu_C12.dat')
O  = load_eos('eos_maxwell_pmu_O16.dat')

# (a) global
fig, ax = plt.subplots(figsize=(6.4, 5.5))
for (n,p,e), c, ls, lab in ((He,CHE,'-',r'$^4$He'), (C,CC,'--',r'$^{12}$C'), (O,CO,'-.',r'$^{16}$O')):
    m = e < 1350
    ax.plot(e[m], p[m], color=c, ls=ls, lw=2.4)
    ax.plot([],[], color=c, ls=ls, lw=2.4, label=lab)
ax.set_xlim(0, 1300); ax.set_ylim(0, 460)
ax.set_xlabel(r'$\varepsilon$ (MeV fm$^{-3}$)'); ax.set_ylabel(r'$p$ (MeV fm$^{-3}$)')
ax.legend(loc='upper left')
fig.savefig(f'{OUT}/fig_eos_pmu_a.png'); plt.close(fig)

# (b) He enlargement (linear, units 1e-6)
fig, ax = plt.subplots(figsize=(6.4, 5.5))
n,p,e = He; S=1e6
m = e <= 0.080
ax.plot(e[m], p[m]*S, color=CHE, lw=2.4)
JH = jumps(*He)
for k,(el,eh,pt) in enumerate(JH[:2],1):
    ax.plot([el,eh],[pt*S,pt*S],'k-',lw=3.4, solid_capstyle='round', zorder=5)
    dy = 0.55 if k==1 else 0.55
    ax.text(0.5*(el+eh), pt*S+dy, rf'$p_t^{{({k})}}$', ha='center', va='bottom', fontsize=15)
ax.set_xlim(0, 0.075); ax.set_ylim(0, 10.4)
ax.set_xlabel(r'$\varepsilon$ (MeV fm$^{-3}$)'); ax.set_ylabel(r'$p$ ($10^{-6}$ MeV fm$^{-3}$)')
fig.savefig(f'{OUT}/fig_eos_pmu_b.png'); plt.close(fig)

# (c),(d): C and O log-log enlargements
for tag, (n,p,e), c, xlim, ylim in (
        ('c', C, CC, (1e-6, 2.0), (3e-11, 1.5e-3)),
        ('d', O, CO, (8e-4, 3.0), (3e-7, 1.2e-3))):
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    m = (e>=xlim[0]*0.5)&(e<=xlim[1]*2)
    ax.plot(e[m], p[m], color=c, lw=2.4)
    JJ = jumps(n,p,e)
    JJ = [j for j in JJ if ylim[0] < j[2] < ylim[1]]
    for k,(el,eh,pt) in enumerate(JJ,1):
        ax.plot([el,eh],[pt,pt],'k-',lw=3.4, solid_capstyle='round', zorder=5)
        ax.text(np.sqrt(el*eh), pt*1.35, rf'$p_t^{{({k})}}$', ha='center', va='bottom', fontsize=14)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\varepsilon$ (MeV fm$^{-3}$)'); ax.set_ylabel(r'$p$ (MeV fm$^{-3}$)')
    fig.savefig(f'{OUT}/fig_eos_pmu_{tag}.png'); plt.close(fig)

print('Fig 2 panels done')

# ---------------- M-R helpers ----------------
def load_mr(name):
    d = np.loadtxt(f'{MRD}/{name}')
    d = d[np.argsort(d[:,0])]
    d = d[d[:,1] < 1e5]           # drop non-converged ultra-low-density rows
    return d

def wd_split(d):
    """Return (stable WD branch, short post-max WD segment)."""
    wd = d[(d[:,1] > 300) & (d[:,1] < 6e4)]
    i = np.argmax(wd[:,2])
    st, po = wd[:i+1], wd[i:]
    Mmax, Rmax = st[-1,2], st[-1,1]
    po = po[(po[:,2] > 0.85*Mmax) & (po[:,1] > 0.3*Rmax) & (po[:,1] < 3*Rmax)]
    return st, po


def _break_gaps(seg, fac=1.8):
    """Insert NaN rows where the nc grid jumps (excluded coexistence gaps),
    so line plots don't draw fake straight connectors."""
    if len(seg) < 3: return seg
    out=[seg[0]]
    for i in range(1, len(seg)):
        if seg[i,0] > seg[i-1,0]*fac:
            out.append([np.nan]*seg.shape[1])
        out.append(seg[i])
    return np.array(out)

def split_full(d):
    """(stable WD, unstable middle, stable NS, post-NS-max) via turning points of M(nc)."""
    d = d[d[:,1] < 1.2e5]; d = d[np.argsort(d[:,0])]
    iw = np.argmax(np.where(d[:,1] > 300, d[:,2], -1))              # WD max
    ns = d[d[:,1] < 120]
    nc_nsmax = ns[np.argmax(ns[:,2]), 0]
    mid = d[(d[:,0] > d[iw,0]) & (d[:,0] < nc_nsmax)]
    nc_min = mid[np.argmin(mid[:,2]), 0]                            # NS minimum mass
    a = d[:iw+1]
    b = d[(d[:,0] >= d[iw,0]) & (d[:,0] <= nc_min)]
    cseg = d[(d[:,0] >= nc_min) & (d[:,0] <= nc_nsmax)]
    e = d[d[:,0] >= nc_nsmax]
    return a, b, cseg, e

def ns_split(d):
    a, b, cseg, e = split_full(d)
    return cseg, e

def obs_overlay(ax, band_label_x=16.8):
    # J0952-0607 mass band 2.35 +- 0.11 (arXiv:2512.05099)
    ax.axhspan(2.24, 2.46, color='0.82', alpha=0.55, zorder=0)
    ax.text(band_label_x, 2.455, r'J0952$-$0607', ha='right', va='top', color='0.35', fontsize=13)
    # J0740+6620 (Salmi 2024)
    ax.errorbar(12.49, 2.073, xerr=[[0.88],[1.28]], yerr=0.069, fmt='D', color='#a03d3d',
                ms=8, capsize=3, lw=1.6, zorder=6)
    ax.text(13.95, 2.073, 'J0740', color='#a03d3d', fontsize=13, ha='left', va='center')
    # J0030+0451 (Kini 2026)
    ax.errorbar(12.68, 1.43, xerr=[[1.04],[1.31]], yerr=[[0.17],[0.20]], fmt='s', color='#8e5bb5',
                ms=8, capsize=3, lw=1.6, zorder=6)
    ax.text(12.15, 1.66, 'J0030', color='#8e5bb5', fontsize=13, ha='center')
    # GW170817 (common-EOS component radii, 90%)
    ax.add_patch(plt.Rectangle((10.5, 1.18), 13.3-10.5, 1.58-1.18, facecolor='#b4c7e7',
                 edgecolor='none', alpha=0.45, zorder=0))
    ax.text(10.65, 1.225, 'GW170817', color='#4a6a9d', fontsize=13)
    # J0437-4715 (Miller 2026)
    ax.errorbar(13.892, 1.431, xerr=[[2.069],[1.250]], yerr=0.043, fmt='^', color='#3a7d44',
                ms=9, capsize=3, lw=1.6, zorder=6)
    ax.text(15.2, 1.36, 'J0437', color='#3a7d44', fontsize=13)

R_sun_km = 6.957e5
_g  = np.loadtxt(f'{OBSD}/MR_Gaia_inBWD.txt')
_h1 = np.loadtxt(f'{OBSD}/MR_Hipparcos_inBWD.txt')
_h2 = np.loadtxt(f'{OBSD}/MR_Hipparcos_DirectlyObservedWD.txt')
_h  = np.vstack([_h1, _h2])
OBS_G = (_g[:,2]*R_sun_km*0.01, _g[:,0], _g[:,3]*R_sun_km*0.01, _g[:,1])
OBS_H = (_h[:,2]*R_sun_km*0.01, _h[:,0], _h[:,3]*R_sun_km*0.01, _h[:,1])

dHe  = load_mr('mr_dense_He.dat')
dC   = load_mr('mr_dense_C12.dat')
dO   = load_mr('mr_dense_O16.dat')
dTM  = load_mr('mr_dense_He_TM2.dat')
dHWW = np.loadtxt(f'{OBSD}/HWW_1968_nonrotating_mr.dat')

# ---------------- Fig 3: NN1 vs TM2 (He-seeded) ----------------
# (a) WD branch
fig, ax = plt.subplots(figsize=(6.4, 5.5))
ax.errorbar(OBS_G[0], OBS_G[1], xerr=OBS_G[2], yerr=OBS_G[3],
            fmt='x', color='0.35', ms=7, lw=1.0, capsize=0, zorder=2, label='Gaia')
ax.errorbar(OBS_H[0], OBS_H[1], xerr=OBS_H[2], yerr=OBS_H[3],
            fmt='+', color='0.6', ms=8, lw=1.0, capsize=0, zorder=2, label='Hipparcos')
for d, c, ls, lab in ((dHe, CC, '-', 'NN1'), (dTM, CTM, '--', 'TM2')):
    st, po = wd_split(d)
    m = st[:,2] > 0.02
    ax.plot(st[m][:,1], st[m][:,2], color=c, ls=ls, lw=2.2, zorder=4)
    ax.plot([],[], color=c, ls=ls, lw=2.2, label=lab)
    ax.plot(st[-1,1], st[-1,2], 'o', color=c, ms=8, zorder=5)
ax.set_xscale('log')
ax.set_xlim(9e2, 2.6e4); ax.set_ylim(0.0, 1.47)
ax.set_xlabel(r'$R$ (km)'); ax.set_ylabel(r'$M$ ($M_\odot$)')
ax.legend(loc='lower left')
fig.savefig(f'{OUT}/fig_mr_tm_a.png'); plt.close(fig)

# (b) NS branch with observations
fig, ax = plt.subplots(figsize=(6.4, 5.5))
obs_overlay(ax)
for d, c, ls, lab in ((dHe, CC, '-', 'NN1'), (dTM, CTM, '--', 'TM2')):
    st, po = ns_split(d)
    ax.plot(st[:,1], st[:,2], color=c, ls=ls, lw=2.2, zorder=4)
    ax.plot(po[:,1], po[:,2], color=c, ls=ls, lw=1.2, alpha=0.35, zorder=3)
    ax.plot(st[-1,1], st[-1,2], 'o', color=c, ms=8, zorder=7)
ax.text(15.85, 2.70, 'TM2', color=CTM, fontsize=14, ha='left')
ax.text(10.70, 2.10, 'NN1', color=CC, fontsize=14)
ax.set_xlim(10, 17); ax.set_ylim(0.0, 2.85)
ax.set_xlabel(r'$R$ (km)'); ax.set_ylabel(r'$M$ ($M_\odot$)')
fig.savefig(f'{OUT}/fig_mr_tm_b.png'); plt.close(fig)
print('Fig 3 panels done')

# ---------------- Fig 4: main M-R (He, C12, O16) ----------------
# (a) WD branch
fig, ax = plt.subplots(figsize=(6.4, 5.5))
ax.errorbar(OBS_G[0], OBS_G[1], xerr=OBS_G[2], yerr=OBS_G[3],
            fmt='x', color='0.35', ms=7, lw=1.0, capsize=0, zorder=2, label='Gaia')
ax.errorbar(OBS_H[0], OBS_H[1], xerr=OBS_H[2], yerr=OBS_H[3],
            fmt='+', color='0.6', ms=8, lw=1.0, capsize=0, zorder=2, label='Hipparcos')
for d, c, ls, lab in ((dHe, CHE, '-', r'$^4$He'), (dC, CC, '--', r'$^{12}$C'), (dO, CO, '-.', r'$^{16}$O')):
    st, po = wd_split(d)
    m = st[:,2] > 0.02
    ax.plot(st[m][:,1], st[m][:,2], color=c, ls=ls, lw=2.2, zorder=4)
    mp = po[:,2] > 0.02
    ax.plot(po[mp][:,1], po[mp][:,2], color=c, ls=ls, lw=1.0, alpha=0.3, zorder=3)
    ax.plot(st[-1,1], st[-1,2], 'o', color=c, ms=8, zorder=6)
    ax.plot([],[], color=c, ls=ls, lw=2.2, label=lab)
ax.set_xscale('log')
ax.set_xlim(7e2, 2.6e4); ax.set_ylim(0.0, 1.47)
ax.set_xlabel(r'$R$ (km)'); ax.set_ylabel(r'$M$ ($M_\odot$)')
ax.legend(loc='lower left', ncol=1, handlelength=1.9)
fig.savefig(f'{OUT}/fig_mr_pmu_a.png'); plt.close(fig)

# (b) NS branch
fig, ax = plt.subplots(figsize=(6.4, 5.5))
obs_overlay(ax)
for d, c, ls in ((dHe, CHE, '-'), (dC, CC, '--'), (dO, CO, '-.')):
    st, po = ns_split(d)
    ax.plot(st[:,1], st[:,2], color=c, ls=ls, lw=2.2, zorder=4)
    ax.plot(po[:,1], po[:,2], color=c, ls=ls, lw=1.2, alpha=0.35, zorder=3)
    ax.plot(st[-1,1], st[-1,2], 'o', color=c, ms=8, zorder=7)
ax.set_xlim(9, 17); ax.set_ylim(0.0, 2.6)
ax.set_xlabel(r'$R$ (km)'); ax.set_ylabel(r'$M$ ($M_\odot$)')
fig.savefig(f'{OUT}/fig_mr_pmu_b.png'); plt.close(fig)
print('Fig 4 panels done')


# ---------------- Fig 5: complete WD-to-NS M-R relation ----------------

fig, ax = plt.subplots(figsize=(8.6, 5.4))
# Historical benchmark behind the present fixed-A sequences.  Hartle & Thorne
# (1968), Table 3, recalculated the nonrotating HWW configurations introduced
# by Harrison et al. (1965).
ax.plot(dHWW[:,1], dHWW[:,2], color='0.38', ls=':', lw=1.5, zorder=1,
        label='HWW (historical)')
for d, c, ls, lab in ((dHe, CHE, '-', r'$^4$He'), (dC, CC, '--', r'$^{12}$C'), (dO, CO, '-.', r'$^{16}$O')):
    a, b, cseg, e = split_full(d)
    ax.plot(a[:,1], a[:,2], color=c, ls=ls, lw=2.2, zorder=4)
    bb = _break_gaps(b)
    ax.plot(bb[:,1], bb[:,2], color=c, ls=ls, lw=1.0, alpha=0.30, zorder=3)
    ax.plot(cseg[:,1], cseg[:,2], color=c, ls=ls, lw=2.2, zorder=4)
    e2 = e[e[:,2] > 0.85*cseg[-1,2]]
    ax.plot(e2[:,1], e2[:,2], color=c, ls=ls, lw=1.0, alpha=0.30, zorder=3)
    ax.plot(a[-1,1], a[-1,2], 'o', color=c, ms=7, zorder=6)
    ax.plot(cseg[-1,1], cseg[-1,2], 'o', color=c, ms=7, zorder=6)
    ax.plot([],[], color=c, ls=ls, lw=2.2, label=lab)
ax.set_xscale('log')
ax.set_xlim(8, 6e4); ax.set_ylim(0.0, 2.5)
ax.set_xlabel(r'$R$ (km)'); ax.set_ylabel(r'$M$ ($M_\odot$)')
ax.text(13, 2.34, 'NS branch', fontsize=14)
ax.text(2.6e3, 1.42, 'WD branch', fontsize=14)
ax.text(220, 0.30, 'unstable', fontsize=13, color='0.45')
ax.legend(loc='upper center', bbox_to_anchor=(0.47, 0.94), ncol=2,
          columnspacing=1.3, handlelength=2.4)
fig.savefig(f'{OUT}/fig_mr_full.png'); plt.close(fig)
print('Fig 5 (full WD-NS) done')
