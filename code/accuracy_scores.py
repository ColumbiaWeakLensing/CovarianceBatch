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
nsim = [200]

#Load the dummy data
data = pd.read_pickle("../data/dummy_data_power.pkl")
assert isinstance(data,Series)

#Load the emulators
emulator = dict()
for configuration in ["coarse","fine"]:
	emulator[configuration] = Emulator.read("../data/emulators/emulator_power_{0}.pkl".format(configuration))

#Parameters to score (coarse and finely sampled)
parameters = dict()
parameters["fine"] = Ensemble.meshgrid({"Om":np.linspace(0.256,0.2635,100),"sigma8":np.linspace(0.795,0.81,100)})
parameters["fine"]["w"] = -1.0

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
	chi2database("../data/scores_power_accuracy.sqlite",parameters["fine"],specs[configuration],table_name="Om_sigma8")
