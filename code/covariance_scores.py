#!/usr/bin/env python
import logging

import lenstools
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator
from lenstools.statistics.database import chi2database

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

#Load emulator and data
emulator = Emulator.read("../data/dummy_emulator_power.pkl")
data = pd.read_pickle("../data/dummy_data_power.pkl")
assert isinstance(data,Series)

#Parameters to score
parameters = Ensemble.meshgrid({"Om":np.linspace(0.2,0.5,100),"sigma8":np.linspace(0.6,1.0,100)})
parameters["w"] = -1.0

#Specifications for chi2database
specs = dict()

#Number of simulations to test
nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]
for n in nsim:
	feature_name = "power_spectrum_{0}sim".format(n)
	specs[feature_name] = dict()
	
	#Emulator
	specs[feature_name]["emulator"] = emulator.combine_features({feature_name:["power_spectrum"]})

	#Data
	d = Series(data.values,index=["l{0}".format(l) for l in range(15)])
	d.add_name(feature_name)
	specs[feature_name]["data"] = d

	#Covariance
	ensemble_nsim = Ensemble(np.load("../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps{0}/power_spectrum_s0.npy".format(n)),columns=d.index)
	specs[feature_name]["data_covariance"] = ensemble_nsim.cov()

#Score the parameters on each of the feature types
chi2database("../data/scores_power.sqlite",parameters,specs)


