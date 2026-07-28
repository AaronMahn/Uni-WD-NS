#!/usr/bin/env python3
"""Dense TOV M-R curves from Maxwell-constructed EOS tables (n, p, e in fm^-3, MeV/fm^3).
Validated against results/mr_pmu_*.dat before producing dense output."""
import numpy as np
from scipy.integrate import solve_ivp
import sys

# constants (cgs-ish -> work in km, Msun, MeV/fm^3)
MeVfm3_to_km2 = 1.3234e-6      # G/c^4 * (1 MeV/fm^3) in km^-2  : e_geom = e * this
# precise: 1 MeV/fm^3 = 1.60218e32 J/m^3; G/c^4 = 8.2627e-45 m/J -> 1.32379e-12 1/m^2 = 1.32379e-6 1/km^2
MeVfm3_to_km2 = 1.32379e-6
Msun_km = 1.47662              # GM_sun/c^2 in km

class EOS:
    def __init__(self, path):
        d = np.loadtxt(path)
        n, p, e = d[:,0], d[:,1], d[:,2]
        good = np.isfinite(n)&np.isfinite(p)&np.isfinite(e)&(n>0)&(p>0)&(e>0)
        n,p,e = n[good],p[good],e[good]
        o = np.argsort(p, kind='stable')
        n,p,e = n[o],p[o],e[o]
        # make p strictly increasing (Maxwell double rows -> nudge)
        for i in range(1,len(p)):
            if p[i] <= p[i-1]:
                p[i] = p[i-1]*(1+1e-12)
        self.n, self.p, self.e = n,p,e
        self.lp, self.le, self.ln = np.log(p), np.log(e), np.log(n)
    def e_of_p(self, p):
        return np.exp(np.interp(np.log(p), self.lp, self.le))
    def p_of_n(self, n):
        return np.exp(np.interp(np.log(n), self.ln, self.lp))
    def e_of_n(self, n):
        return np.exp(np.interp(np.log(n), self.ln, self.le))

def tov_solve(eos, p_c, p_surf, rtol=1e-8):
    """Integrate TOV from center; returns R [km], M [Msun]."""
    ec = eos.e_of_p(p_c)
    # geometric units: eg = e*K, pg = p*K  [km^-2]
    K = MeVfm3_to_km2
    r0 = 1e-6
    m0 = (4/3)*np.pi*r0**3 * ec*K      # km
    def rhs(r, y):
        pg, m = y
        if pg <= 0: return [0,0]
        e = eos.e_of_p(pg/K)*K
        dp = -(e+pg)*(m + 4*np.pi*r**3*pg)/(r*(r-2*m))
        dm = 4*np.pi*r**2*e
        return [dp, dm]
    def hit_surf(r, y): return y[0] - p_surf*K
    hit_surf.terminal = True; hit_surf.direction = -1
    sol = solve_ivp(rhs, [r0, 5e5], [p_c*K, m0], events=hit_surf,
                    rtol=rtol, atol=[p_surf*K*1e-3, 1e-12], max_step=np.inf,
                    method='RK45', first_step=1e-4)
    if sol.t_events[0].size:
        R = sol.t_events[0][0]; M = sol.y_events[0][0][1]
    else:
        R = sol.t[-1]; M = sol.y[1][-1]
    return R, M/Msun_km

if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    eosHe = EOS(f'{base}/eosdata/eos_maxwell_pmu_He.dat')
    # validation points
    ref = np.loadtxt(f'{base}/results/mr_pmu_He_refined_ns.dat')
    for psurf in (1e-18, 1e-16, 1e-14, 1e-12):
        out=[]
        for nc in (0.55, 0.7, 1.0):
            pc = eosHe.p_of_n(nc)
            R,M = tov_solve(eosHe, pc, psurf)
            out.append(f'nc={nc}: R={R:.3f} M={M:.4f}')
        i0 = np.argmin(abs(ref[:,0]-0.55)); i1=np.argmin(abs(ref[:,0]-0.7)); i2=np.argmin(abs(ref[:,0]-1.0))
        print(f'p_surf={psurf:.0e}:', ' | '.join(out))
        print('   ref:', ' | '.join(f'nc={ref[i,0]:.3f}: R={ref[i,1]:.3f} M={ref[i,2]:.4f}' for i in (i0,i1,i2)))
