#!/usr/bin/env python
# coding: utf-8

# ## Eccentric Residual Search Ideal

# This notebook tests on a simulated dataset containing an eccentric gw signal. Based on work done by Sarah Vigeland, Ph.D. from `cw_search_sample.ipynb`
# 
# Updated: 07/1/2020

# In[1]:


from __future__ import division
import numpy as np
import glob
import os
import pickle
import json
import matplotlib.pyplot as plt
import corner
import sys

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise_extensions.sampler import JumpProposal as JP
import enterprise.constants as const
from enterprise.signals import utils
from enterprise_extensions.deterministic import CWSignal
from enterprise_extensions.deterministic import cw_block_circ
from enterprise.signals.signal_base import SignalCollection
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import ecc_res
import scipy.constants as sc


def get_noise_from_pal2(noisefile):
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params


#Get data and noise files
datadir = '../../ecc_signal_create/ecc_sim_data/ideal_data/logmc_10.5/closer_dist/quasi_circ/'
noisepath = '/home/bcheeseboro/nanograv_proj/enterprise_proj/ecc_signal_create/small_pta_noise
noisefiles = sorted(glob.glob(noisepath+'/*.txt'))

#Load pulsars from pickle file
picklefile = 'ideal_pulsars_ecc_search.pkl'
filename = datadir + picklefile

with open(filename, "rb") as f:
    psrs = pickle.load(f)


# white noise parameter
efac = parameter.Constant()
selection = selections.Selection(selections.by_backend)

# white noise signal
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

#Eccentric gw parameters
#gw parameters
gwphi = parameter.Constant(5.02)('gwphi')
gwtheta = parameter.Constant(2.51)('gwtheta')
log10_dist = parameter.Constant(6.0)('log10_dist')

#orbital parameters
l0 = parameter.Constant(0)('l0') #mean anomaly
gamma0 = parameter.Constant(0)('gamma0') #initial angle of periastron
inc = parameter.Constant(np.pi/3)('inc') #inclination of the binary's orbital plane
psi = parameter.Constant(0)('psi') #polarization of the GW

#Search parameters
q = parameter.Uniform(0.1,1)('q') #mass ratio
log10_mc = parameter.Uniform(7,11)('log10_mc') #log10 chirp mass
e0 = parameter.Uniform(0.001, 0.1)('e0') #eccentricity
log10_forb = parameter.Uniform(-9,-7)('log10_forb')#log10 orbital frequency
#p_dist = parameter.Normal(0,1) #pulsar distance
#pphase = parameter.Uniform(0,2*np.pi) #pulsar phase


#Eccentric signal construction
#To create a signal to be used by enterprise you must first create a residual 
#and use CWSignal to convert the residual as part of the enterprise Signal class
ewf = ecc_res.add_ecc_cgw(gwtheta=gwtheta, gwphi=gwphi, log10_mc=log10_mc, q=q, log10_forb=log10_forb, e0=e0, l0=l0, gamma0=gamma0, inc=inc, psi=psi, log10_dist=log10_dist, p_dist=None, pphase=None, gamma_P=None, tref=59000, psrterm=True, evol=True, waveform_cal=True, res='Both')

ew = CWSignal(ewf, ecc=False, psrTerm=False)

# linearized timing model
tm = gp_signals.TimingModel(use_svd=False)
# full signal (no red noise added at this time)
s = ef + tm + ew


# initialize PTA
model = [s(psr) for psr in psrs]
pta = signal_base.PTA(model)

#add noise params
params = {}
for nf in noisefiles:
    params.update(get_noise_from_pal2(nf))

#Set default pta params
pta.set_default_params(params)

#Set up walker position
xecc = np.hstack(np.array([p.sample() for p in pta.params]))
ndim = len(xecc)


# initialize pulsar distance parameters
p_dist_params = [ p for p in pta.param_names if 'p_dist' in p ]
for pd in p_dist_params:
    xecc[pta.param_names.index(pd)] = 0

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

#group specific params together
groups = [range(0, 4), [0,1], [1,2], [2,3]]

chaindir = '../ecc_search_data/ideal_data_test/detection_runs/closer_dist/quasi_circ/run2/'


resume = True
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, 
                 outDir=chaindir, resume=resume)

# write parameter file and parameter groups file
np.savetxt(chaindir + 'params.txt', list(map(str, pta.param_names)), fmt='%s')
np.savetxt(chaindir + 'groups.txt', groups, fmt='%s')

# add prior draws to proposal cycle
jp = JP(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 25)

#run sampler
N = int(1.5e6)
sampler.sample(xecc, N, SCAMweight=40, AMweight=20, DEweight=60)

