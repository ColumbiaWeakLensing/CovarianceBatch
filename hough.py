#!/usr/bin/env python-mpi

import sys,os
import logging

from lenstools import ConvergenceMap
from lenstools.index import *
from lenstools.statistics import Ensemble
from lenstools.pipeline import SimulationBatch

import numpy as np
from mpi4py import MPI

from emcee.utils import MPIPool

#Hough transform
from skimage.transform import hough_line

##########################################################################################################################
##############This function reads in a ConvergenceMap and measures all the descriptors provided in index##################
##########################################################################################################################

def convergence_hough(filename,threshold,bins,mean_subtract):

	"""
	Measures all the statistical descriptors of a convergence map as indicated by the index instance
	
	"""

	logging.info("Processing {0}".format(filename))

	#Load the map
	conv_map = ConvergenceMap.load(filename)

	if mean_subtract:
		conv_map.data -= conv_map.mean()

	#Compute the hough transform
	linmap = conv_map.data > np.random.rand(*conv_map.data.shape) * threshold
	out,angle,d = hough_line(linmap)

	#Compute the histogram of the hough transform map
	h,b = np.histogram(out.flatten()*1.0/linmap.sum(),bins=bins)

	#Return
	return h


##########################################################################################################################
##############This function measures all the descriptors in a map set#####################################################
##########################################################################################################################

def measure_from_set(filename,map_set,threshold,bins,mean_subtract=False):
	return map_set.execute(filename,callback=convergence_hough,threshold=threshold,bins=bins,mean_subtract=mean_subtract)


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
	threshold = 0.1
	bins = np.linspace(0.0,0.0014,50)
	np.save("bins_hough.npy",bins)

	#How much
	num_realizations = 1024

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring descriptors for simulation batch at {0}".format(batch.environment.home))

	#Get a handle on the collection
	model = batch.getModel(model_id)
	collection = model.getCollection(box_size=240.0*model.Mpc_over_h,nside=512)

	#Perform the measurements for all the map sets
	for map_set in collection.mapsets[:1]:

		#Log to user
		logging.info("Processing map set {0}".format(map_set.settings.directory_name))

		#Construct an ensemble for each map set
		ensemble = Ensemble.fromfilelist([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(num_realizations) ])

		#Measure the descriptors spreading calculations on a MPIPool
		ensemble.load(callback_loader=measure_from_set,pool=pool,map_set=map_set,threshold=threshold,bins=bins)

		#Save
		savename = os.path.join(map_set.home_subdir,"hough.npy")
		logging.info("Writing {0}".format(savename))
		ensemble.save(savename)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





