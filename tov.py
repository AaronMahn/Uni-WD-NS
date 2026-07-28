#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

PYTOV:
======

    PYTOV is a simple Python Tolman-Oppenheimer-Volkoff (TOV) equation 
    integrator. 
    
    Summary:
    ========    
    -   A high density equation of state file and a low density equation of 
        state file will be read and then combined in a simple manner. 
    -   Logarithmic interpolation of the combined equation of state is used. 
    -   A simple fixed step fourth order Runge-Kutta method is used to 
        integrate the TOV equations. A central density is specified and the 
        TOV equations are integrated out to the surface of the compact star. 
        This is repeated for a range of densities. 
    -   The required data for a mass vs radius curve is output to a file along
        with a file containing the details of the maximum mass compact star.
        
@author: D. L. Whittenbury
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
# Useful constants

# Conversion factors (see Norman K. Glendenning, Compact Stars: Nuclear Physics, Particle Physics and General Relativity)
MeVfm_3Tokm_2 = 1.3234e-6  # 1 MeVfm^-3 = 1.3234e-6 km^-2
km_2ToMeVfm_3 = 1.0 / MeVfm_3Tokm_2
ModotTokm = 1.4766  # 1 M_odot = 1.4766 km
kmToModot = 1.0 / ModotTokm




@jit
def density(P,neos,peos,eeos):
    """Calculate the baryonic density as a function of pressure

    INPUT: Pressure [km^-2]

    OUTPUT: Baryonic density [fm^-3 ]

    NOTES: Only interpolation and no extrapolation.

    """
    if P < 0:
        return 0

    else:

        i = np.argmax(peos > P)
        # Finds the index of the first p in peos > P

        if i == 0:
            return neos[0]  # No extrapolation to lower density/pressure used

        else:

            last = len(neos) - 1
            # The last index

            if i <= last:  # Then interpolate
                den = neos[i]*np.exp(np.log(P/peos[i])*np.log(neos[i-1]/neos[i]) /np.log(peos[i-1]/peos[i]));
                #den = neos[i] * (P / peos[i])* (neos[i - 1] / neos[i])/ (peos[i - 1] / peos[i])
                

                return den

            else:
                return neos[last]  # No extrapolation to higher density/pressure

@jit
def energydensity(P,neos,peos,eeos):

    """Calculate the energy density as a function of pressure

    INPUT: Pressure [km^-2]

    OUTPUT: Energy density [km^-2]

    NOTES: Only logarithmic interpolation and no extrapolation.

    """
    if P < 0:
        return 0
    else:

        i = np.argmax(peos > P)
        # Finds the index of the first p in peos > P

        if i == 0:
            return eeos[0]  # No extrapolation to lower en. density/pressure used

        else:

            last = len(neos) - 1
            # The last index

            if i <= last:  # Then interpolate

                eden = eeos[i] * np.exp(
                    np.log(P / peos[i])
                    * np.log(eeos[i - 1] / eeos[i])
                    / np.log(peos[i - 1] / peos[i])
                )
                #eden = eeos[i] * (P / peos[i])* (eeos[i - 1] / eeos[i])/ (peos[i - 1] / peos[i])
                
                return eden

            else:
                return eeos[
                    last
                ]  # No extrapolation to higher en. density/pressure used

@jit
def pressure(n,neos,peos,eeos):
    """Calculate the pressure as a function of baryonic density

    INPUT: Baryonic density [fm^-3 ]

    OUTPUT: Pressure [km^-2]

    NOTES: Only logarithmic interpolation and no extrapolation.

    """
    if n < 0:
        return 0

    else:

        i = np.argmax(neos > n)
        # Finds the index of the first n in neos > n

        if i == 0:
            return peos[0]  # No extrapolation to lower density/pressure used

        else:

            last = len(neos) - 1
            # The last index

            if i <= last:  # Then interpolate

                press = peos[i] * np.exp(
                    np.log(n / neos[i])
                    * np.log(peos[i - 1] / peos[i])
                    / np.log(neos[i - 1] / neos[i])
                )
                #press = peos[i] * (n / neos[i])* (peos[i - 1] / peos[i])/ (neos[i - 1] / neos[i])
                


                return press

            else:
                return peos[last]  # No extrapolation to higher density/pressure used

@jit
def TOVderivs(y, t,neos,peos,eeos):
    """Calculate the derivatives in the TOV equations

    INPUT:  Dependent variables y = y(t), where y is gravitational mass M,
            graviational field Phi, pressure P and baryon number of baryons A.
            The independent variable is r.

    OUTPUT: Derivatives of dependent variables mass M, graviational field Phi,
            pressure P and baryon number of baryons A.

    NOTES:  Uses the functions density() and energydensity() which utilise
            logarithmic interpolation.

    """

    # Radius in km
    r = t
    r2 = r**2
    r3 = r**3

    # The dependent variables y
    M = y[0]  # Mass in km , note 1 solar mass = 1.4766 km

    # Phi = y[1]; # Gravitational potential, later match to suface

    P = y[2]  # Pressure km^-2

    # A = y[3] # Number of baryons

    # Lookup energy density and baryonic density
    nB = density(P,neos,peos,eeos)  # fm^-3
    E = energydensity(P,neos,peos,eeos)  # km^-2

    # Array of derivative equations
    dydx = np.zeros(4)

    # Eq. for graviational mass
    dydx[0] = 4.0 * np.pi * r2 * E

    # Gravitational potential - to be continued ... Needs to be matched!!
    dydx[1] = (M + 4.0 * np.pi * r3 * P) / (r2 * (1.0 - 2.0 * M / r))

    # Hydrostatic equilibrium ... equation for pressure
    dydx[2] = -(E + P) * dydx[1]

    # Eq. for baryon number A
    dydx[3] = (4.0 * np.pi * r2 * nB) / np.sqrt(1.0 - 2.0 * M / r)

    return dydx

@jit
def RK4Step(u, t, dt, derivs,neos,peos,eeos):
    """Fourth order Runge Kutta method

    INPUT:  Dependent variables u, independent variable t, step size dt,
            function to calculate derivatives of dependent variables
            with respect to t, i.e., expect derivs(u,t).

    OUTPUT: Advanced independent variables unew.

    NOTES:  See Applied numerical methods for engineers and scientists,
            S. S. Rao, page 659.

    """

    n = len(u)
    unew = np.zeros(n)
    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)

    k1 = derivs(u, t,neos,peos,eeos)
    k2 = derivs(u + 0.5 * dt * k1, t + 0.5 * dt,neos,peos,eeos)
    k3 = derivs(u + 0.5 * dt * k2, t + 0.5 * dt,neos,peos,eeos)
    k4 = derivs(u + dt * k3, t + dt,neos,peos,eeos)

    unew = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return unew

@jit
def TOV_initial(rs: float, nB: float,neos,peos,eeos):
    """
    Constructs the initial vector y for the dependent variables, gravitational
    mass M, graviational field Phi, pressure P and baryon number of baryons A.

    rs: Small radial offset [km].
    nB: Total baryonic density [fm^-3].

    """

    # Dependent variables
    y = np.zeros(4)

    # We are not at the centre of the star, but rather slightly off centre,
    # starting at rs.
    # Gravitational mass M[r=0] = 0, but we are not at the centre so we have a
    # small contribution
    y[0] = (4.0 * np.pi / 3.0) * energydensity(pressure(nB,neos,peos,eeos),neos,peos,eeos) * rs**3

    # Graviational field Phi
    # Phi[r=0] is not known. Actually phi will be matched at the surface, so we
    # could set the y[1] = 0 and there would be no difference! BUT we can also
    # choose phi0 = 0 at the centre and add a second term which is the
    # correction because we are not starting at the centre.
    y[1] = (
        0.0
        + (2.0 * np.pi / 3.0)
        * (energydensity(pressure(nB,neos,peos,eeos),neos,peos,eeos) + 3.0 * pressure(nB,neos,peos,eeos))
        * rs**2
    )

    # Pressrue P[r=0] = P0, but we are not at the centre, so we have a small
    # additional contribution
    y[2] = pressure(nB,neos,peos,eeos)
    -(2.0 * np.pi / 3.0) * (energydensity(pressure(nB,neos,peos,eeos),neos,peos,eeos) + pressure(nB,neos,peos,eeos)) * (
        energydensity(pressure(nB,neos,peos,eeos),neos,peos,eeos) + 3.0 * pressure(nB,neos,peos,eeos)
    ) * rs**2

    # Number of baryons A[r=0] = A0 = 0 , but we are not at the centre so we
    # have a small number of baryons
    y[3] = (4.0 * np.pi / 3.0) * nB * rs**3

    return y

@jit
def IntegrateTOV(uStart, tStart, dt, derivs,neos,peos,eeos):
    """Integrate the TOV equations

    INPUT:  Starting values for the dependent variables uStart, starting
            value for the dependent variable tStart, step size dt, the
            function to evaluate derivatives of the dependent variables u
            with respect to the independent variable t, i.e., expect
            derivs(u,t).

    OUTPUT: The variable out which contains (R, MG, Phi, P, Ab),
            radius R, gravitational mass MG, graviational field Phi (but
            not matched yet), pressure P (when integration stopped)),
            number of baryons Ab.

    NOTES:

    """

    # Maximum no. of steps
    step_limit = 1000000

    n = len(uStart)

    out = np.zeros(n + 1)  # r and n variables

    Var = np.zeros(n)
    Var = uStart  # Starting variable values

    r = tStart  # Starting r
    dr = dt  # Step size

    # Solutions to be stored here
    sol = np.zeros((step_limit, n))
    sol[0] = Var
    counter = 0
    while sol[counter, 2] > peos[0] and counter < step_limit - 1:
        # Stopping criteria: Continue while pressure is greater than the smallest
        #                    value in the EoS table and while the no. of steps is
        #                    less than the chosen limit. No extrapolation to lower
        #                    values of pressure, we take the smallest value as
        #                    approx. defining the location of the surface.
        deltaVar=10
#         while deltaVar>0.01:
        newVar = RK4Step(Var, r, dr, derivs,neos,peos,eeos)  # RK4 step forward
        deltaVar=np.sum(np.abs(newVar-Var))
        Var=newVar
#         print(r)
        counter = counter + 1
        sol[counter] = Var
        r = r + dr

    if sol[counter, 2] < 0:  # Do not want a negative pressure solution so subtract 1
        counter = counter - 1

    # out = (r, sol[counter,0], sol[counter,1], sol[counter,2], sol[counter,3]) =(R, MG,Phi, P, Ab)
    out[0] = r
    out[1 : n + 1] = sol[counter]
    return out,sol

@jit
def determine_max_mass(
    masses: np.ndarray, radii: np.ndarray, rhocs: np.ndarray, phis: np.ndarray
):
    """
    A simple determination of the maximum mass.

    One could of course be more accurate and interpolate and find a maximum.

    """

    # Maximum mass
    mass_max = max(masses)

    # Index of maximum mass neutron star
    idx = [i for i, v in enumerate(masses) if v == mass_max]

    # The corresponding radius
    radius_max = (radii[idx])[0]

    # The corresponding central density
    rho_c_max = rhocs[idx][0]

    # The gravitational field on the surface
    phi_max = (phis[idx])[0]

    return mass_max, radius_max, rho_c_max, phi_max


@jit
def main(eos,iterationss,stepsize):
    iterations=int(iterationss)
    # PREPARE THE EQUATION OF STATE(S)
    # ================================

    # Reads in EoS. If order is not specified, then it is assumed to be
    # density, pressure and energy density. read_eos outputs the eos in the
    # default order density, pressure, energy density.
    

    # When you combine EoS they should be in same order and have the same units. Unit conversion is on you, the user.
    combined_eos = eos


    # Separate EoS and convert units if necessary for solving the TOV equations. Unit conversion is on you, the user.
    # Density should be fm^-3
    # Pressure and energy density should be km^-2
    neos = combined_eos[:, 0]
    peos = combined_eos[:, 1] * MeVfm_3Tokm_2
    eeos = combined_eos[:, 2] * MeVfm_3Tokm_2

    # Select a central density and integrate out to the surface. Repeat for a
    # range of densities. This will give you the required data for a mass vs
    # radius curve.

    # SOLVE THE TOV EQUATIONS
    # =======================

    # Saturation density
    

    # Starting density
    startingDensity = 0.00001

    # Stepsize in density
    #stepsize = 0.02 * n0

    # central density loop
    # Number of iterations (0.8*0.16 + 0.02*0.16*335 = 1.2 [fm^-3])
    #iterations = 750

    # Start off centre
    rs = 0.00001  # [km]
    # Stepsize (APPROX. TIME: dr = 0.01 about ~1 min, dr = 0.001 about ~10 min )
    # Best to use 0.001 or smaller when you require accurate results. It all
    # depends on how long you are willing to wait. Although, you could improve this
    # integrator by upgrading to a variable step size integrator.
    dr = 0.001  # 0.01; 0.001; 0.0001;

    radii = np.zeros(iterations)
    masses = np.zeros(iterations)
    phis = np.zeros(iterations)
    rhocs = np.zeros(iterations)
    # Loop over densities
    data=np.zeros([iterations,3,1000000]);
    for i in range(iterations):
        # Central density

        nB = startingDensity + stepsize * float(i)

        # Central density
        rhocs[i] = nB

        y = TOV_initial(rs, nB,neos,peos,eeos)

        # Solutions (r, M, Phi (not matched yet), P, A)
        solution,sol = IntegrateTOV(y, rs, dr, TOVderivs,neos,peos,eeos)
        sol[:,0]=sol[:,0]* kmToModot;
        data[i,0,0:len(sol[:,0])]=sol[:,0];
        data[i,1,0:len(sol[:,2])]=sol[:,2];
        data[i,2,0:len(sol[:,3])]=sol[:,3];
        # Convert to correct units before output to file
        radii[i] = solution[0]
        masses[i] = solution[1] * kmToModot
        phis[i] = solution[2]

        # Not yet matched at the surface
        # Rescale phi to match at surface with Schwarzchild solution
        const = 0.5 * np.log(1.0 - 2.0 * masses[i] * ModotTokm / radii[i]) - phis[i]
        phis[i] = phis[i] + const

        Pre = solution[3] * km_2ToMeVfm_3
        Ab = solution[4] * 1.0e54  # fm^-3 <-> km^-3

        # Print to screen as calculation progresses
        print(i, nB, radii[i], masses[i])

    # Determine maximum mass configuration
    mass_max, radius_max, rho_c_max, phi_max = determine_max_mass(
        masses=masses, radii=radii, rhocs=rhocs, phis=phis
    )

    return radii,masses,rhocs,data