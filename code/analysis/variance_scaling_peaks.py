#!/usr/bin/env python
from __future__ import division

import sys
sys.modules["mpi4py"] = None

from operator import add

from lenstools.statistics.database import Database
from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.constraints import FisherAnalysis

import numpy as np
from scipy.sparse import csr_matrix

import algorithms

############################################################
#Compression matrix (49x15) to go from 49 to 15 kappa bands#
############################################################

i = reduce(add,[(n,n,n) for n in range(15)])
j = np.arange(45)
d = np.ones(45) / 3.
compression = csr_matrix((d,(i,j)),shape=(15,49)).T.toarray()

###########################
######Main execution#######
###########################

def main():

	#Number of simulations to test
	nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]
	nreal = np.arange(100,1100,100)
	resample = 100

	#Load the fisher analysis
	fisher = FisherAnalysis.read("../../data/fisher_peaks.pkl").refeaturize(compression,method="dot")
	feature_columns = fisher[fisher.feature_names].columns

	#Load in the feature Ensemble, and bootstrap the covariance using a different number of realizations
	with Database("../../data/variance_scaling_expected_peaks.sqlite") as db:

		#Log
		table_name = "variance_15bands"
		print("[+] Populating table {0}...".format(table_name))

		#This is the reference covariance matrix
		true_covariance = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps200/peaks_s0.npy").dot(compression),columns=feature_columns).cov()
		diagonal_covariance = Ensemble(np.diag(true_covariance.values.diagonal()),index=true_covariance.index,columns=true_covariance.columns)

		for n in nsim:
		
			#Load the power spectrum Ensemble from the relevant map set
			ensemble_nsim = Ensemble(np.load("../../Om0.260_Ol0.740_w-1.000_ns0.960_si0.800/512b240/Maps{0}/peaks_s0.npy".format(n)).dot(compression),columns=feature_columns)

			#Bootstrap ensemble_sim and compute the parameter variance for each resample
			for nr in nreal:
				print("[+] Bootstraping peak constraints, Nb={0}, nsim={1}, nreal={2} with {3} resamples".format(ensemble_nsim.shape[1],n,nr,resample))
				variance_ensemble = ensemble_nsim.bootstrap(algorithms.bootstrap_fisher,bootstrap_size=nr,resample=resample,assemble=lambda l:Ensemble.concat(l,ignore_index=True),fisher=fisher,true_covariance=true_covariance,extra_items={"nsim":n,"nreal":nr})
				db.insert(Ensemble(variance_ensemble.mean()).T,table_name=table_name)

if __name__=="__main__":
	main()
