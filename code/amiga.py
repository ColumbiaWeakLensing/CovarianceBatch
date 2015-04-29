import os,glob

from lenstools.pipeline.simulation import SimulationBatch
from lenstools import Ensemble

import numpy as np
import astropy.units as u

def read_amiga_txt(f):
	return np.loadtxt(f,unpack=True)

def read_mass(d):
	return d[3]

def scale_mass(r,m):
	return m*u.Msun/r.cosmology.h

#######################################################################
###################General loader######################################
####################################################################### 

def read_halo_stats(n,snapshot_number,model="Om0.260_Ol0.740_w-1.000_ns0.960_si0.800",collection="512b240",reader=read_amiga_txt,callback=read_mass,post_callback=scale_mass):

	"""
	Read a statistic from the halo finder

	"""
	
	#Simulation batch is the current one
	batch = SimulationBatch.current()

	#Get the particular IC
	ic = batch.getModel(model).getCollection(collection).getRealization(n)

	#Construct the filename that contains the halo statistics
	halo_root_filename = os.path.join(ic.path("amiga"),"snap{0}".format(snapshot_number))
	halo_filenames = glob.glob(halo_root_filename+"*AHF_halos")

	#Cumulate the desired property from all the files
	statistic = list()
	for halo_file in halo_filenames:
		print("[+] Reading {0}...".format(halo_file))
		data_in_file = reader(halo_file)
		data_subset = callback(data_in_file)
		statistic.append(data_subset)

	#hstack all
	print("[+] Merging...")
	statistic = np.hstack(statistic)

	#Additional post--processing
	statistic = post_callback(ic,statistic)

	#Done, return
	return statistic

#######################################################################
###################Convenience#########################################
#######################################################################

def read_redshift(n,snapshot_number,model="Om0.260_Ol0.740_w-1.000_ns0.960_si0.800",collection="512b240"):

	"""
	Read the redshift from the filename

	"""

	#Simulation batch is the current one
	batch = SimulationBatch.current()

	#Get the particular IC
	ic = batch.getModel(model).getCollection(collection).getRealization(n)

	#Construct the filename that contains the halo statistics
	halo_root_filename = os.path.join(ic.path("amiga"),"snap{0}".format(snapshot_number))
	halo_filenames = glob.glob(halo_root_filename+"*AHF_halos")

	#Read the redshift from the first filename
	parts = os.path.basename(halo_filenames[0]).split(".")
	z_int = parts[-3].strip("z")
	z_float = parts[-2]

	#Return
	return float("{0}.{1}".format(z_int,z_float))




