#!/usr/bin/env python
from __future__ import division
import gc

from lenstools.statistics.database import Database
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import FisherAnalysis,Emulator

import numpy as np

import algorithms

###########################
######Main execution#######
###########################

#Number of simulations to test
nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]
nreal = np.arange(100,1100,100)
resample = 100

def main():

	############################################################
	####Constraints from the power spectrum (linear binning)####
	############################################################

	#Load the emulator
	emulator = Emulator.read("../../data/emulators/emulator_power_fine_nb39.pkl")
	num_ell = np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/ell_nb39.npy")
	feature_columns = emulator[emulator.feature_names].columns

	#Large and small scales
	scales = {
	"large" : filter(lambda s:int(s.strip("l")) in range(15),emulator[emulator.feature_names[0]].columns),
	"small" : filter(lambda s:int(s.strip("l")) in range(15,30),emulator[emulator.feature_names[0]].columns),
	"large+small" : filter(lambda s:int(s.strip("l")) in range(30),emulator[emulator.feature_names[0]].columns),
	"all" : list(emulator[emulator.feature_names[0]].columns),
	}

	#Fiducial cosmology
	fiducial_parameters = Series(np.array([0.26,-1.0,0.8]),index=["Om","w","sigma8"])

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../../data/variance_scaling_gaussian_expected.sqlite") as db:

		#This is the true covariance matrix
		true_covariance = Ensemble(np.diag((emulator.predict(fiducial_parameters).values**2)/num_ell),index=feature_columns,columns=feature_columns)
	
		for s in scales:

			#Cut the number of bins
			sub_feature_names = [(emulator.feature_names[0],fb) for fb in scales[s]]

			#Log
			table_name = "power_" + s
			print("[+] Populating table {0}...".format(table_name))

			#Approximate the emulator linearly around the fiducial model
			fisher_scale = emulator.approximate_linear(fiducial_parameters,derivative_precision=0.01).features({emulator.feature_names[0]:scales[s]})
			true_covariance_scale = true_covariance[sub_feature_names].loc[sub_feature_names]

			for n in nsim:
		
				#Mock the power spectrum Ensemble drawing from a multivariate Gaussian
				ensemble_nsim = Ensemble.sample_gaussian(true_covariance_scale,realizations=1000,mean=emulator.predict(fiducial_parameters)[sub_feature_names])

				#Bootstrap ensemble_sim and compute the parameter variance for each resample
				for nr in nreal:

					gc.collect()

					print("[+] Bootstraping scale={0}, Nb={1}, nsim={2}, nreal={3} with {4} resamples".format(s,ensemble_nsim.shape[1],n,nr,resample))
					variance_ensemble = ensemble_nsim.bootstrap(algorithms.bootstrap_fisher,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),fisher=fisher_scale,true_covariance=true_covariance_scale,extra_items={"nsim":n,"nreal":nr,"bins":ensemble_nsim.shape[1]})
					db.insert(Ensemble(variance_ensemble.mean()).T,table_name=table_name)


	###########################################################
	####Constraints from the power specrum (log binning)#######
	###########################################################

	#Load the emulator
	emulator = FisherAnalysis.read("../../data/emulators/emulator_power_fine.pkl")
	num_ell = np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/ell.npy")
	feature_columns = emulator[emulator.feature_names].columns

	#Large and small scales
	scales = {
	"large" : filter(lambda s:int(s.strip("l"))<=7,emulator[emulator.feature_names[0]].columns),
	"small" : filter(lambda s:int(s.strip("l"))>7,emulator[emulator.feature_names[0]].columns),
	"all" :  list(emulator[emulator.feature_names[0]].columns),
	}

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../../data/variance_scaling_gaussian_expected.sqlite") as db:

		#This is the true covariance matrix
		true_covariance = Ensemble(np.diag((emulator.predict(fiducial_parameters).values**2)/num_ell),index=feature_columns,columns=feature_columns)
	
		for s in scales:

			#Cut the number of bins
			sub_feature_names = [(emulator.feature_names[0],fb) for fb in scales[s]]

			#Log
			table_name = "power_logb_" + s
			print("[+] Populating table {0}...".format(table_name))

			#Approximate the emulator linearly around the fiducial model
			fisher_scale = emulator.features({emulator.feature_names[0]:scales[s]})
			true_covariance_scale = true_covariance[sub_feature_names].loc[sub_feature_names]

			for n in nsim:
		
				#Mock the power spectrum Ensemble drawing from a multivariate Gaussian
				ensemble_nsim = Ensemble.sample_gaussian(true_covariance_scale,realizations=1000,mean=emulator.predict(fiducial_parameters)[sub_feature_names])

				#Bootstrap ensemble_sim and compute the parameter variance for each resample
				for nr in nreal:

					gc.collect()

					print("[+] Bootstraping scale={0}, Nb={1}, nsim={2}, nreal={3} with {4} resamples".format(s,ensemble_nsim.shape[1],n,nr,resample))
					variance_ensemble = ensemble_nsim.bootstrap(algorithms.bootstrap_fisher,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),fisher=fisher_scale,true_covariance=true_covariance_scale,extra_items={"nsim":n,"nreal":nr,"bins":ensemble_nsim.shape[1]})
					db.insert(Ensemble(variance_ensemble.mean()).T,table_name=table_name)


if __name__=="__main__":
	main()
