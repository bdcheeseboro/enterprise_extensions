from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
import numpy as np
from numpy import pi, sin, cos
import scipy.constants as sc


import antenna_pattern as ap
import ecc_utils as eu
import waveform as waveform

GMsun = 1.327124400e20  # measured more precisely than Msun alone!
c = sc.speed_of_light
TSUN = GMsun / c**3
parsec = sc.parsec
KPC2S = parsec/c * 1e3

@signal_base.function
def add_ecc_cgw(toas, theta, phi, pdist = 1.0, gwtheta, gwphi, log10_dist = 6, log10_mc, q, log10_fgw, e0, l0, gamma0, inc, psi, l_P = None, gamma_P = None, tref = 0, psrterm = True, evol = True, waveform_cal = True, res = 'Both'):
    """
    Simulate GW from eccentric SMBHB. Waveform models from
    Susobhanan et al. (2020).
    This residual waveform is very accurate even if the
    GW frequency is significantly evolving over the 
    observation time of the pulsar.
    :param toas: array of time of arrivals that come from the pulsar object
    :param theta: polar coordinate of pulsar position [radians]
    :param phi: azimuthal coordinate of pulsar position [radians]
    :param pdist: pulsar distance [Kpc]
    :param gwtheta: polar coordinate of gravitational wave source position [radians]
    :param gwphi: azimuthal coordinate of gravitational wave source position [radians]
    :param log10_dist: Base-10 luminosity distance to gravitational wave source
    :param log10_mc: Base-10 of the chirp mass of the binary
    :param q: Mass ratio of the SMBHB [dimensionless]
    :param log10_fgw: Base-10 of twice the orbital frequency of SMBHB
    :param e0: Initial eccentricity of SMBHB
    :param l0: Initial mean anomaly [radians]
    :param gamma0: Initial angle of periastron [radians]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param tref: Fiducial time at which initial parameters are referenced [MJd]
    :param psrterm: Whether to include the pulsar term into the calculations [Boolean]
    :waveform_cal: Whether to add the numerical calculation to the waveform [Boolean]
    :parameter evol: Whether to evolve the binary during the observation time window [Boolean]
    :param res: Which term to use for calculating residual [string]. Use 'Earth' for only earth term, 'Pulsar' for only pulsar term & 'Both' for addig both the terms
    :returns: Vector of induced residuals
    """
    F0 = 10**log10_fgw
    mc = 10**log10_mc
    gwdist = 10**log10_dist
    n0 = 2 * pi * F0
    tref = tref * 86400
    
	
    cosmu, Fp, Fx = ap.antenna_pattern(gwphi, gwtheta, phi, theta)
	
    if (res == 'Both' or res == 'Earth'):
        #print("Calculating Earth term")
        residual = - waveform.calculate_sp_sx(toas, gwdist, mc, q, n0, e0, l0, gamma0, inc, psi, tref, Fp, Fx, evol, waveform_cal)

	
    #if psrterm:
    if (res == 'Both' or res == 'Pulsar'):
        #print("Calculating Pulsar term")
        Dp = pdist * KPC2S
        dt_P = Dp * (1 - cosmu)

        toas_P = toas - dt_P
        tref_P = tref - dt_P

        n0_P, e0_P, l0_P, gamma0_P = eu.evolve_orbit(tref_P, mc, q, n0, e0, l0, gamma0, tref)

        if l_P is not None:
            l0_P = l_P
        if gamma_P is not None:
            gamma0_P = gamma_P

        residual_P = waveform.calculate_sp_sx(toas_P, pdist, mc, q, n0_P, e0_P, l0_P, gamma0_P, inc, psi, tref_P, Fp, Fx, evol, waveform_cal)


        if res == 'Pulsar':
            residual = residual_P/86400
        elif res == 'Both':
            residual = (residual_P + residual)/86400
    return residual