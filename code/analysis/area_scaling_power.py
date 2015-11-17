#!/usr/bin/env python
from __future__ import division

from lenstools.statistics.database import Database
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator

import numpy as np
import pandas as pd

import algorithms

###########################
######Main execution#######
###########################

def main():

	#Number of simulations to test
	nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]
	nreal = np.arange(100,1100,100)
	resample = 100

	#Load the emulator and the test data
	emulator = Emulator.read("../../data/test/test_emulator.pkl")
	feature_columns = emulator[emulator.feature_names].columns
	test_data = pd.read_pickle("../data/test/test_power.pkl")

	#Test data
	fiducial_parameters = Series(np.array([0.26,-1.0,0.8]),index=["Om","w","sigma8"])

	#Parameters on which to comute the scores
	parameter_grid = Ensemble.meshgrid({'Om':np.linspace(0.15,0.5,100),'sigma8':np.linspace(0.5,1.0,100)})
	parameter_grid["w"] = -1.

	#Large and small scales
	scales = {
	"all" : emulator[emulator.feature_names[0]].columns, 
	}

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../../data/area_scaling.sqlite") as db:
	
		for s in scales:

			#Downsize emulator and test data
			emulator_scale = emulator.features({"power_spectrum":scales[s]})
			test_data_scale = test_data[ [("power_spectrum",ell) for ell in scales[s]] ] 

			#Approximate the emulator linearly around the fiducial model
			fisher = emulator_scale.approximate_linear(fiducial_parameters,derivative_precision=0.01)

			#This is the reference covariance matrix
			true_covariance_ensemble = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps200/power_spectrum_s0.npy"),columns=emulator[emulator.feature_names[0]].columns)[scales[s]]
			true_covariance = true_covariance_ensemble.cov()
			diagonal_covariance = Ensemble(np.diag(true_covariance.values.diagonal()),index=true_covariance.index,columns=true_covariance.columns)

			for n in nsim:
		
				#Load the power spectrum Ensemble from the relevant map set
				ensemble_nsim = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps{0}/power_spectrum_s0.npy".format(n)),columns=emulator[emulator.feature_names[0]].columns)[scales[s]]
				ensemble_nsim.add_name("power_spectrum")

				#Bootstrap ensemble_sim and compute the parameter variance for each resample
				for nr in nreal:
					print("[+] Bootstraping scale={0}, nsim={1}, nreal={2} with {3} resamples".format(s,n,nr,resample))
					area_ensemble = ensemble_nsim.bootstrap(algorithms.bootstrap_area,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),emulator=emulator_scale,parameter_grid=parameter_grid,test_data=test_data_scale,extra_items={"nsim":n,"nreal":nr},pool=None)
					db.insert(Ensemble(area_ensemble),table_name="Om_si_area_"+s)


if __name__=="__main__":
	main()
