#!/usr/bin/env python

import lenstools
from lenstools.simulations import Design,Nicaea
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator

import numpy as np

#Multipoles to emulate
ell = np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/ell.npy")

#Emulate the power spectrum for each of the parameters, both in the coarse and fine configurations
for configuration in ["fine"]:

	#Load the designs
	parameters = Design.read("../../data/designs/design_{0}.pkl".format(configuration))

	#Emulate the power spectrum with Nicaea
	emulated_power = np.zeros((len(parameters),len(ell)))
	for n,(Om,w,sigma8) in enumerate(parameters.values):
		print("[+] Emulating {0} configuration {1} of {2}...".format(configuration,n+1,len(parameters)))
		cosmo = Nicaea(Om0=Om,Ode0=1.-Om,w0=w,sigma8=sigma8)
		emulated_power[n] = cosmo.convergencePowerSpectrum(ell,z=2.0)

	#Construct the emulator
	parameters.add_name("parameters")
	emulated_power = Ensemble(emulated_power,columns=["l{0}".format(n) for n in range(len(ell))])
	emulated_power.add_name("power_spectrum")

	#Save the emulator
	Emulator.from_features(emulated_power,parameters=parameters).to_pickle("../../data/emulators/emulator_power_{0}.pkl".format(configuration))

#Construct the dummy data
cosmo = Nicaea(Om0=0.26,w0=-1.0,sigma8=0.8)
data = Series(cosmo.convergencePowerSpectrum(ell,z=2.0),index=["l{0}".format(n) for n in range(len(ell))])
data.to_pickle("../../data/dummy_data_power.pkl")
