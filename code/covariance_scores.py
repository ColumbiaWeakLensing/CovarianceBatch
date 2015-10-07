#!/usr/bin/env python
import logging

import lenstools
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator
from lenstools.statistics.database import chi2database

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

#Number of simulations to test
nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]

#Load the dummy data
data = pd.read_pickle("../data/dummy_data_power.pkl")
assert isinstance(data,Series)

#Load the emulators
emulator = dict()
for configuration in ["coarse","fine"]:
	emulator[configuration] = Emulator.read("../data/emulators/emulator_power_{0}.pkl".format(configuration))

#Parameters to score (coarse and finely sampled)
parameters = dict()
parameters["coarse"] = Ensemble.meshgrid({"Om":np.linspace(0.2,0.5,100),"w":np.linspace(-1.2,-0.8,100)})
parameters["coarse"]["sigma8"] = 0.8
parameters["fine"] = Ensemble.meshgrid({"Om":np.linspace(0.256,0.2635,100),"w":np.linspace(-1.08,-0.92,100)})
parameters["fine"]["sigma8"] = 0.8


#Specifications for chi2database
specs = dict()

for configuration in ["coarse","fine"]:

	specs[configuration] = dict()

	for n in nsim:
		
		feature_name = "{0}_{1}sim".format(configuration,n)
		specs[configuration][feature_name] = dict()
	
		#Emulator
		specs[configuration][feature_name]["emulator"] = emulator[configuration].combine_features({feature_name:["power_spectrum"]})

		#Data
		d = data.copy()
		d.add_name(feature_name)
		specs[configuration][feature_name]["data"] = d

		#Covariance
		ensemble_nsim = Ensemble(np.load("../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps{0}/power_spectrum_s0.npy".format(n)),columns=d.index)
		specs[configuration][feature_name]["data_covariance"] = ensemble_nsim.cov() / 1600

	#Score the parameters on each of the feature types
	chi2database("../data/scores_power_LSST.sqlite",parameters[configuration],specs[configuration],table_name=configuration+"_Om_w")
