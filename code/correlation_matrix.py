#!/usr/bin/env python-mpi

from __future__ import division

import sys,os
import logging

from lenstools import ConvergenceMap
from lenstools.statistics import Ensemble
from lenstools.pipeline import SimulationBatch

import numpy as np
from mpi4py import MPI

from emcee.utils import MPIPool

##########################################################################################################################
##############This function reads in a ConvergenceMap and measures all the descriptors provided in index##################
##########################################################################################################################

def cross_correlation(pair,map_set,redshift,index2realization):

	"""
	Measures the cross correlation between two convergence maps
	
	"""

	#Read in the maps
	map1 = ConvergenceMap.load(os.path.join(map_set.storage_subdir,"WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,index2realization[pair[0]]+1)))
	map2 = ConvergenceMap.load(os.path.join(map_set.storage_subdir,"WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,index2realization[pair[1]]+1)))

	#Compute the cross correlation
	return (map1.data * map2.data).sum() / np.sqrt((map1.data**2).sum() * (map2.data**2).sum())

#################################################################################
##############Main execution#####################################################
#################################################################################

if __name__=="__main__":

	logging.basicConfig(level=logging.INFO)

	#Initialize MPIPool
	try:
		pool = MPIPool()
	except:
		pool = None

	if (pool is not None) and not(pool.is_master()):
		
		pool.wait()
		pool.comm.Barrier()
		MPI.Finalize()
		sys.exit(0)

	#How many realizations
	num_realizations = 10

	#Allocate space for the correlation matrix
	correlation_matrix = np.zeros((num_realizations,)*2)

	#Upper diagonal indices of a matrix represent all the pairs to measure
	pair_indices = np.triu_indices(num_realizations,1)
	pairs = np.array(pair_indices).T

	#Translate matrix index into realization number
	index2realization = np.arange(num_realizations)
	label = "correlation_matrix"

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring map cross correlation for simulation batch at {0}".format(batch.environment.home))

	#Get a handle on the map set
	model_id = "Om0.260_Ol0.740_w-1.000_ns0.960_si0.800"
	redshift = 2.0
	model = batch.getModel(model_id)
	collection = model.getCollection(box_size=240.0*model.Mpc_over_h,nside=512)
	map_set = collection.getMapSet("10000Maps1sim")

	#Log to user
	logging.info("Processing map set {0}".format(map_set.settings.directory_name))

	#Construct an ensemble for each map set
	ensemble_pairs = Ensemble.fromfilelist(pairs)
	ensemble_pairs.load(cross_correlation,pool=pool,map_set=map_set,redshift=redshift,index2realization=index2realization)

	#Fill the entries of the correlation matrix
	correlation_matrix[pair_indices] = ensemble_pairs.data

	#Complete the matrix
	correlation_matrix += correlation_matrix.T
	correlation_matrix[np.diagonal_indices(num_realizations)] = 1.0

	#Save the correlation matrix
	savename = os.path.join(map_set.home_subdir,label+".npy")
	logging.info("Saving correlation matrix to {0}".format(savename))
	np.save(savename,correlation_matrix)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





