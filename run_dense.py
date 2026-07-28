import numpy as np, sys
from multiprocessing import Pool
from tov_dense import EOS, tov_solve

import os
base = os.path.dirname(os.path.abspath(__file__))
PSURF = 1e-17

def jumps_of(eos, thr=1.04):
    J=[]
    for i in range(1,len(eos.p)):
        if eos.p[i] <= eos.p[i-1]*(1+1e-9) and eos.n[i] > eos.n[i-1]*thr:
            J.append((eos.n[i-1], eos.n[i]))
    return J

def make_grid(eos, n_lo=1e-8, n_hi=None, N=450):
    n_hi = n_hi or eos.n[-1]*0.999
    g = np.logspace(np.log10(n_lo), np.log10(max(n_lo*10,n_hi)), N)
    J = jumps_of(eos)
    ok = np.ones(g.size, bool)
    for a,b in J:
        ok &= ~((g>a*0.999)&(g<b*1.001))
    return g[ok]

def solve_one(args):
    path, nc = args
    eos = EOS_CACHE[path]
    try:
        R,M = tov_solve(eos, eos.p_of_n(nc), PSURF)
        return (nc, R, M)
    except Exception:
        return (nc, np.nan, np.nan)

EOS_CACHE={}
def run(path, out, n_hi=None):
    eos = EOS(path); EOS_CACHE[path]=eos
    grid = make_grid(eos, n_hi=n_hi)
    with Pool(10) as pool:
        res = pool.map(solve_one, [(path,nc) for nc in grid])
    arr = np.array([r for r in res if np.isfinite(r[1])])
    # refine near local maxima of M(nc)
    for _ in range(2):
        M = arr[:,2]; add=[]
        for i in range(1,len(arr)-1):
            if M[i]>M[i-1] and M[i]>M[i+1]:
                add += list(np.geomspace(arr[i-1,0], arr[i+1,0], 14)[1:-1])
        J = jumps_of(eos); good=[]
        for nc in add:
            if not any(a*0.999<nc<b*1.001 for a,b in J): good.append(nc)
        if not good: break
        with Pool(10) as pool:
            res = pool.map(solve_one, [(path,nc) for nc in good])
        arr = np.vstack([arr, [r for r in res if np.isfinite(r[1])]])
        arr = arr[np.argsort(arr[:,0])]
    np.savetxt(out, arr, header='nc[fm^-3] R[km] M[Msun]  (dense TOV, p_surf=4e-12 MeV/fm^3)')
    i=np.argmax(arr[:,2]); print(out, len(arr), f'Mmax={arr[i,2]:.4f} at R={arr[i,1]:.2f} nc={arr[i,0]:.3f}')
    # also WD max (R>500km)
    wd = arr[arr[:,1]>500]
    if len(wd): i=np.argmax(wd[:,2]); print('   WD max:', f'{wd[i,2]:.4f} Msun at {wd[i,1]:.1f} km')

if __name__=='__main__':
    run(f'{base}/eosdata/eos_maxwell_pmu_He.dat',      f'{base}/results/mr_dense_He.dat')
    run(f'{base}/eosdata/eos_maxwell_pmu_C12.dat',     f'{base}/results/mr_dense_C12.dat')
    run(f'{base}/eosdata/eos_maxwell_pmu_O16.dat',     f'{base}/results/mr_dense_O16.dat')
    run(f'{base}/eosdata/eos_maxwell_pmu_He_TM2.dat',  f'{base}/results/mr_dense_He_TM2.dat')
