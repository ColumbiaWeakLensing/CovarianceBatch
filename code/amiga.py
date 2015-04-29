import os,glob

from lenstools.pipeline.simulation import SimulationBatch
from lenstools import Ensemble

import astropy.units as u

def read_amiga_txt(f):
	return np.loadtxt(f,unpack=True)

def read_mass(r,d):
	return d[3]*u.Msun / r.cosmology.h

def read_halo_stats(n,snapshot_number,model="Om0.260_Ol0.740_w-1.000_ns0.960_si0.800",collection="512b240",reader=read_amiga_txt,callback=read_mass):
	
	#Simulation batch is the current one
	batch = SimulationBatch.current()

	#Get the particular IC
	ic = batch.getModel(model).getCollection(collection).getRealization(n)

	#Construct the filename that contains the halo statistics
	halo_root_filename = os.path.join(ic.path("amiga"),"snap{0}".format(snapshot_number))
	halo_filenames = glob.glob(halo_root_filename+"*AHF_halos")

	#Check
	return halo_filenames