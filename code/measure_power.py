#!/usr/bin/env python-mpi
from __future__ import division

import sys,os
import logging

from lenstools.image.convergence import ConvergenceMap
from lenstools.statistics.ensemble import Ensemble
from lenstools.pipeline import SimulationBatch

import numpy as np
import astropy.units as u
from mpi4py import MPI

from emcee.utils import MPIPool

#############################################################################################
##############Measure the power spectrum#####################################################
#############################################################################################

def convergence_power(fname,map_set,l_edges,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		l,Pl = conv.powerSpectrum(l_edges)
		return Pl

	except IOError:
		return None

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

	#Where 
	model_id = "Om0.260_Ol0.740_w-1.000_ns0.960_si0.800"
	redshift = 2.0

	#What to measure
	l_edges = np.logspace(2.0,np.log10(6.0e3),16)

	#How many realizations
	num_realizations = 1024
	chunks = 16
	realizations_per_chunk = num_realizations // chunks

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring descriptors for simulation batch at {0}".format(batch.environment.home))

	#Get a handle on the collection
	model = batch.getModel(model_id)
	collection = model.getCollection(box_size=240.0*model.Mpc_over_h,nside=512)

	#Save for reference
	np.save(os.path.join(collection.home_subdir,"ell_nb{0}.npy".format(len(l_edges)-1)),0.5*(l_edges[1:]+l_edges[:-1]))

	#Perform the measurements for all the map sets
	for map_set in collection.mapsets[:-2]:

		#Log to user
		logging.info("Processing map set {0}".format(map_set.settings.directory_name))

		#Construct an ensemble for each map set
		ensemble_all = list()

		#Measure the descriptors spreading calculations on a MPIPool
		for c in range(chunks):
			ensemble_all.append(Ensemble.compute([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(realizations_per_chunk*c,realizations_per_chunk*(c+1)) ],callback_loader=convergence_power,pool=pool,map_set=map_set,l_edges=l_edges))

		#Merge all the chunks
		ensemble_all = Ensemble.concat(ensemble_all,axis=0,ignore_index=True)

		#Save to disk
		savename = os.path.join(map_set.home_subdir,"power_spectrum_s{0}_nb{1}.npy".format(0,ensemble_all.shape[1]))
		logging.info("Writing {0}".format(savename))
		np.save(savename,ensemble_all.values)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





