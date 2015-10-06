#!/usr/bin/env python

import lenstools
from lenstools.simulations import Nicaea
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator

import numpy as np

#Cosmological parameters and multipoles to build the power spectrum emulator
parameters = np.load(lenstools.data("CFHTemu1_array.npy"))
ell = np.load("../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/ell.npy")

#Emulate the power spectrum for each of the parameters
emulated_power = np.zeros((len(parameters),len(ell)))
for n,(Om,w,sigma8) in enumerate(parameters):
	cosmo = Nicaea(Om0=Om,w0=w,sigma8=sigma8)
	emulated_power[n] = cosmo.convergencePowerSpectrum(ell,z=2.0)

#Construct the emulator
parameters = Ensemble(parameters,columns=["Om","w","sigma8"])
parameters.add_name("parameters")
emulated_power = Ensemble(emulated_power,columns=["l{0}".format(n) for n in range(len(ell))])
emulated_power.add_name("power_spectrum")

Emulator.from_features(emulated_power,parameters=parameters).to_pickle("../data/dummy_emulator_power.pkl")

#Construct the dummy data
cosmo = Nicaea(Om0=0.26,w0=-1.0,sigma8=0.8)
data = Series(cosmo.convergencePowerSpectrum(ell,z=2.0),index=["l{0}".format(n) for n in range(len(ell))])
data.add_name("power_spectrum")
data.to_pickle("../data/dummy_data_power.pkl")
