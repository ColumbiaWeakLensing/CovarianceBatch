#!/usr/bin/env python
from __future__ import division
import gc

from lenstools.statistics.database import Database
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator

import numpy as np

import algorithms

###########################
######Main execution#######
###########################

def main():

	#Number of simulations to test
	nsim = [1]
	nreal = [100,500,1000] + range(2000,101000,1000)
	resample = 100

	#Load the emulator
	emulator = Emulator.read("../../data/emulators/emulator_power_fine.pkl")
	feature_columns = emulator[emulator.feature_names].columns

	#Large and small scales
	scales = {
	"large" : filter(lambda s:int(s.strip("l"))<=7,emulator[emulator.feature_names[0]].columns),
	"small" : filter(lambda s:int(s.strip("l"))>7,emulator[emulator.feature_names[0]].columns),
	"all" :  list(emulator[emulator.feature_names[0]].columns),
	}

	#Fiducial cosmology
	fiducial_parameters = Series(np.array([0.26,-1.0,0.8]),index=["Om","w","sigma8"])

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../../data/variance_scaling_power_largeNr.sqlite") as db:
	
		for s in scales:

			#Log
			table_name = "power_logb_" + s
			print("[+] Populating table {0}...".format(table_name))

			#Approximate the emulator linearly around the fiducial model
			fisher = emulator.approximate_linear(fiducial_parameters,derivative_precision=0.01).features({"power_spectrum":scales[s]})

			#This is the reference covariance matrix
			true_covariance_ensemble = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/GrandEnsemble/power_spectrum_s0.npy"),columns=emulator[emulator.feature_names[0]].columns)[scales[s]]
			true_covariance = true_covariance_ensemble.cov()
			diagonal_covariance = Ensemble(np.diag(true_covariance.values.diagonal()),index=true_covariance.index,columns=true_covariance.columns)

			for n in nsim:
		
				#Load the power spectrum Ensemble from the relevant map set
				ensemble_nsim = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/MillionMapsPower/power_spectrum_s0.npy".format(n)),columns=emulator[emulator.feature_names[0]].columns)[scales[s]]
				ensemble_nsim.add_name("power_spectrum")

				#Bootstrap ensemble_sim and compute the parameter variance for each resample
				for nr in nreal:

					gc.collect()

					print("[+] Bootstraping scale={0}, Nb={1}, nsim={2}, nreal={3} with {4} resamples".format(s,ensemble_nsim.shape[1],n,nr,resample))
					variance_ensemble = ensemble_nsim.bootstrap(algorithms.bootstrap_fisher,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),fisher=fisher,true_covariance=true_covariance,extra_items={"nsim":n,"nreal":nr,"bins":ensemble_nsim.shape[1]})
					db.insert(Ensemble(variance_ensemble.mean()).T,table_name=table_name)


if __name__=="__main__":
	main()
