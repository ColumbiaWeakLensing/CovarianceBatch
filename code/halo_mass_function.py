#!/usr/bin/env python

import os

from amiga import read_halo_stats,read_redshift

from lenstools.pipeline.simulation import SimulationBatch
from lenstools import Ensemble

import numpy as np
import astropy.units as u

##########################################################################
#########Read the text file and select the mass column####################
##########################################################################

def read_amiga_txt(f):
	return np.loadtxt(f,unpack=True)

def read_mass(d):
	return d[3]

######################################################
#########Compute the mass function####################
######################################################

def mass_function(r,m):
	
	#Binning
	bins = np.logspace(10.0,15.0,51)

	#Make the histogram
	h,b = np.histogram(m,bins=bins)

	#Convert the histogram into dn/dm
	return h / ((b[1:]-b[:-1])*r.box_size.value**3)



def main():

	#keyword arguments to pass to read_halo_stats
	kwargs = {
	
	"snapshot_number" : 46,
	"model" : "Om0.260_Ol0.740_w-1.000_ns0.960_si0.800",
	"collection" : "512b240",
	"reader" : read_amiga_txt,
	"select" : read_mass, 
	"post_process" : mass_function 
	
	}

	#Useful handler
	collection = SimulationBatch.current().getModel(kwargs["model"]).getCollection(kwargs["collection"])

	#These realizations will make up the ensemble
	ics_in_ensemble = range(101,201)

	#Read the redshift
	z = read_redshift(ics_in_ensemble[0],kwargs["snapshot_number"],kwargs["model"],kwargs["collection"])

	#Build the ensemble
	ens = Ensemble.fromfilelist(ics_in_ensemble)
	ens.load(read_halo_stats,**kwargs)

	#Binning
	bins = np.logspace(10.0,15.0,51)
	savename = os.path.join(collection.home_subdir,"mass.npy")
	print("[+] Saving mass bins to {0}".format(savename))
	np.save(savename,0.5*(bins[1:]+bins[:-1]))

	#Save the ensemble
	savename = os.path.join(collection.home_subdir,"mass_function_z{0:.3f}.npy".format(z))
	print("[+] Saving mass function to {0}".format(savename))
	ens.save(savename)

if __name__=="__main__":
	main()