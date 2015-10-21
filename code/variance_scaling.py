#!/usr/bin/env python

from lenstools.statistics.database import Database
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import Emulator

import numpy as np

###############################
###Bootstrap fisher ellipses###
###############################

def bootstrap_fisher(ensemble,fisher,true_covariance,extra_items):
	
	pcov = fisher.parameter_covariance(ensemble.cov(),observed_features_covariance=true_covariance)
	pvar = Series(pcov.values.diagonal(),index=pcov.index)
	for key in extra_items.keys():
		pvar[key] = extra_items[key]
	
	return pvar.to_frame().T


###########################
######Main execution#######
###########################

def main():

	#Number of simulations to test
	nsim = [1,100]
	nreal = np.arange(50,650,10)
	resample = 100

	#Load the emulators
	emulator = dict()
	feature_columns = dict()

	for configuration in ["fine"]:
		emulator[configuration] = Emulator.read("../data/emulators/emulator_power_{0}.pkl".format(configuration))
		feature_columns[configuration] = emulator[configuration][emulator[configuration].feature_names].columns

	#Fiducial cosmology
	fiducial_parameters = Series(np.array([0.26,-1.0,0.8]),index=["Om","w","sigma8"])

	#This is the reference covariance matrix
	true_covariance = Ensemble(np.load("../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps200/power_spectrum_s0.npy"),columns=feature_columns["fine"]).cov()

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../data/variance_scaling.sqlite") as db:
	
		for configuration in ["fine"]:
			for n in nsim:
		
				#Load the power spectrum Ensemble from the relevant map set
				ensemble_nsim = Ensemble(np.load("../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps{0}/power_spectrum_s0.npy".format(n)),columns=feature_columns["fine"])

				#Approximate the emulator linearly around the fiducial model
				fisher = emulator[configuration].approximate_linear(fiducial_parameters,derivative_precision=0.01)

				#Bootstrap ensemble_sim and compute the parameter variance for each resample
				for nr in nreal:
					print("[+] Bootstraping configuration={0}, nsim={1}, nreal={2} with {3} resamples".format(configuration,n,nr,resample))
					variance_ensemble = ensemble_nsim.bootstrap(bootstrap_fisher,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),fisher=fisher,true_covariance=true_covariance,extra_items={"nsim":n,"nreal":nr,"configuration":configuration})
					db.insert(variance_ensemble,table_name="variance")


if __name__=="__main__":
	main()
