from __future__ import division,with_statement

import sys,os
import logging
import time
import cPickle
import resource

from distutils import config

from lenstools.utils import MPIWhirlPool

from lenstools import ConvergenceMap
from lenstools import Ensemble

from lenstools.simulations.raytracing import RayTracer
from lenstools.pipeline.simulation import SimulationBatch
from lenstools.pipeline.settings import MapSettings

import numpy as np
import astropy.units as u

################################################
###########Loggers##############################
################################################

console = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)-12s:%(levelname)-4s: %(message)s",datefmt='%m-%d %H:%M')
console.setFormatter(formatter)

logdriver = logging.getLogger("lenstools.driver")
logdriver.addHandler(console)
logdriver.propagate = False

#Orchestra director of the execution
def powerSpectrumExecution():

	script_to_execute = singleRedshift
	settings_handler = PowerSpectrumSettings
	kwargs = {}

	return script_to_execute,settings_handler,kwargs

################################################
#######Single redshift ray tracing##############
################################################

def singleRedshift(pool,batch,settings,id,**kwargs):

	#Safety check
	assert isinstance(pool,MPIWhirlPool) or (pool is None)
	assert isinstance(batch,SimulationBatch)

	parts = id.split("|")

	if len(parts)==2:

		assert isinstance(settings,PowerSpectrumSettings)
	
		#Separate the id into cosmo_id and geometry_id
		cosmo_id,geometry_id = parts

		#Get a handle on the model
		model = batch.getModel(cosmo_id)

		#Get the corresponding simulation collection and map batch handlers
		collection = [model.getCollection(geometry_id)]
		map_batch = collection[0].getMapSet(settings.directory_name)
		cut_redshifts = np.array([0.0])

	elif len(parts)==1:

		assert isinstance(settings,PowerSpectrumSettings)

		#Get a handle on the model
		model = batch.getModel(parts[0])

		#Get the corresponding simulation collection and map batch handlers
		map_batch = model.getTelescopicMapSet(settings.directory_name)
		collection = map_batch.mapcollections
		cut_redshifts = map_batch.redshifts

	else:
		
		if (pool is None) or (pool.is_master()):
			logdriver.error("Format error in {0}: too many '|'".format(id))
		sys.exit(1)


	#Override the settings with the previously pickled ones, if prompted by user
	if settings.override_with_local:

		logdriver.error("Cannot override with local!!")
		sys.exit(1)

	##################################################################
	##################Settings read###################################
	##################################################################

	#Set random seed to generate the realizations
	if pool is not None:
		np.random.seed(settings.seed + pool.rank)
	else:
		np.random.seed(settings.seed)

	#Read map angle,redshift and resolution from the settings
	map_angle = settings.map_angle
	source_redshift = settings.source_redshift
	resolution = settings.map_resolution

	if len(parts)==2:

		#########################
		#Use a single collection#
		#########################

		#Read the plane set we should use
		plane_set = (settings.plane_set,)

		#Randomization
		nbody_realizations = (settings.mix_nbody_realizations,)
		cut_points = (settings.mix_cut_points,)
		normals = (settings.mix_normals,)
		map_realizations = settings.lens_map_realizations

	elif len(parts)==1:

		#######################
		#####Telescopic########
		#######################

		#Check that we have enough info
		for attr_name in ["plane_set","mix_nbody_realizations","mix_cut_points","mix_normals"]:
			if len(getattr(settings,attr_name))!=len(collection):
				if (pool is None) or (pool.is_master()):
					logdriver.error("You need to specify a setting {0} for each collection!".format(attr_name))
				sys.exit(1)

		#Read the plane set we should use
		plane_set = settings.plane_set

		#Randomization
		nbody_realizations = settings.mix_nbody_realizations
		cut_points = settings.mix_cut_points
		normals = settings.mix_normals
		map_realizations = settings.lens_map_realizations



	#Decide which map realizations this MPI task will take care of (if pool is None, all of them)
	if pool is None:
		first_map_realization = 0
		last_map_realization = map_realizations
		realizations_per_task = map_realizations
		logdriver.debug("Generating lensing map realizations from {0} to {1}".format(first_map_realization+1,last_map_realization))
	else:
		assert map_realizations%(pool.size+1)==0,"Perfect load-balancing enforced, map_realizations must be a multiple of the number of MPI tasks!"
		realizations_per_task = map_realizations//(pool.size+1)
		first_map_realization = realizations_per_task*pool.rank
		last_map_realization = realizations_per_task*(pool.rank+1)
		logdriver.debug("Task {0} will generate lensing map realizations from {1} to {2}".format(pool.rank,first_map_realization+1,last_map_realization))

	#Planes will be read from this path
	plane_path = os.path.join("{0}","ic{1}","{2}")

	if (pool is None) or (pool.is_master()):
		for c,coll in enumerate(collection):
			logdriver.info("Reading planes from {0}".format(plane_path.format(coll.storage_subdir,"-".join([str(n) for n in nbody_realizations[c]]),plane_set[c])))

	#Plane info file is the same for all collections
	if (not hasattr(settings,"plane_info_file")) or (settings.plane_info_file is None):
		info_filename = os.path.join(plane_path.format(collection[0].storage_subdir,nbody_realizations[0][0],plane_set[0]),"info.txt")
	else:
		info_filename = settings.plane_info_file

	if (pool is None) or (pool.is_master()):
		logdriver.info("Reading lens plane summary information from {0}".format(info_filename))

	#Read how many snapshots are available
	with open(info_filename,"r") as infofile:
		num_snapshots = len(infofile.readlines())

	#Save path for the maps
	save_path = map_batch.storage_subdir

	if (pool is None) or (pool.is_master()):
		logdriver.info("Lensing maps will be saved to {0}".format(save_path))

	begin = time.time()

	#Allocate space for the power spectrum Ensemble
	ps_ensemble = Ensemble.fromdata(np.zeros(realizations_per_task,settings.l_edges.shape[0]-1))

	#We need one of these for cycles for each map random realization
	for rloc,r in enumerate(range(first_map_realization,last_map_realization)):

		#Instantiate the RayTracer
		tracer = RayTracer()

		start = time.time()
		last_timestamp = start

		#############################################################
		###############Add the lenses to the system##################
		#############################################################

		#Open the info file to read the lens specifications (assume the info file is the same for all nbody realizations)
		infofile = open(info_filename,"r")

		#Read the info file line by line, and decide if we should add the particular lens corresponding to that line or not
		for s in range(num_snapshots):

			#Read the line
			line = infofile.readline().strip("\n")

			#Stop if there is nothing more to read
			if line=="":
				break

			#Split the line in snapshot,distance,redshift
			line = line.split(",")

			snapshot_number = int(line[0].split("=")[1])
		
			distance,unit = line[1].split("=")[1].split(" ")
			if unit=="Mpc/h":
				distance = float(distance)*model.Mpc_over_h
			else:
				distance = float(distance)*getattr(u,"unit")

			lens_redshift = float(line[2].split("=")[1])

			#Select the right collection
			for n,z in enumerate(cut_redshifts):
				if lens_redshift>=z:
					c = n

			#Randomization of planes
			nbody = np.random.randint(low=0,high=len(nbody_realizations[c]))
			cut = np.random.randint(low=0,high=len(cut_points[c]))
			normal = np.random.randint(low=0,high=len(normals[c]))

			#Log to user
			logdriver.debug("Realization,snapshot=({0},{1}) --> NbodyIC,cut_point,normal=({2},{3},{4})".format(r,s,nbody_realizations[c][nbody],cut_points[c][cut],normals[c][normal]))

			#Add the lens to the system
			logdriver.info("Adding lens at redshift {0}".format(lens_redshift))
			plane_name = os.path.join(plane_path.format(collection[c].storage_subdir,nbody_realizations[c][nbody],plane_set[c]),"snap{0}_potentialPlane{1}_normal{2}.fits".format(snapshot_number,cut_points[c][cut],normals[c][normal]))
			tracer.addLens((plane_name,distance,lens_redshift))

		#Close the infofile
		infofile.close()

		now = time.time()
		logdriver.info("Plane specification reading completed in {0:.3f}s".format(now-start))
		last_timestamp = now

		#Rearrange the lenses according to redshift and roll them randomly along the axes
		tracer.reorderLenses()

		now = time.time()
		logdriver.info("Reordering completed in {0:.3f}s".format(now-last_timestamp))
		last_timestamp = now

		#Start a bucket of light rays from a regular grid of initial positions
		b = np.linspace(0.0,map_angle.value,resolution)
		xx,yy = np.meshgrid(b,b)
		pos = np.array([xx,yy]) * map_angle.unit

		#Trace the ray deflections
		jacobian = tracer.shoot(pos,z=source_redshift,kind="jacobians")

		now = time.time()
		logdriver.info("Jacobian ray tracing for realization {0} completed in {1:.3f}s".format(r+1,now-last_timestamp))
		last_timestamp = now

		#Compute shear,convergence and omega from the jacobians
		if settings.convergence:
			
			#Compute the convergence
			convMap = ConvergenceMap(data=1.0-0.5*(jacobian[0]+jacobian[3]),angle=map_angle)

			#Measure the power spectrum
			ell,P_ell = convMap.powerSpectrum(settings.l_edges)

			#Save the multipoles maybe
			if r==0:
				savename = os.path.join(map_batch.home_subdir,"ell.npy")
				logdriver.info("Saving multipoles to {0}".format(savename))
				np.save(savename,ell)

			#Add the measured power spectrum to the Ensemble
			ps_ensemble.data[rloc] = P_ell

		else:
			logdriver.error("You need to enable the 'convergence' option!")
			sys.exit(1)
			

		now = time.time()
		logdriver.info("Weak lensing calculations for realization {0} completed in {1:.3f}s".format(r+1,now-last_timestamp))
		logdriver.info("Memory usage: {0:.3f} GB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)))
	
	#Safety sync barrier
	if pool is not None:
		pool.comm.Barrier()

	#Save the local processor Ensemble to disk
	if pool is not None:
		savename = os.path.join(map_batch.home_subdir,"{0}_rank{1:03d}.npy".format(settings.ensemble_root,pool.rank))
	else:
		savename = os.path.join(map_batch.home_subdir,"{0}.npy".format(settings.ensemble_root))

	logdriver.info("Saving power spectrum Ensemble to {0}".format(savename))
	ps_ensemble.save(savename)

	if (pool is None) or (pool.is_master()):	
		now = time.time()
		logdriver.info("Total runtime {0:.3f}s".format(now-begin))


############################################################################################################################################################################

########################################################
###########PowerSpectrumSettings class##################
########################################################

class PowerSpectrumSettings(MapSettings):

	_section = "ConvergencePowerSpectrum"

	@classmethod 
	def read(cls,config_file):

		settings = super(PowerSpectrumSettings,cls).read(config_file)

		#Read the options from the ini file
		options = config.ConfigParser()
		options.read([config_file])

		#Read in the name of the ensemble
		settings.ensemble_root = options.get(cls._section,"ensemble_root")

		#Read in the multipoles
		settings._read_multipoles(options,cls._section)

		#Return
		return settings

	def _read_multipoles(self,options,section):
		
		self.lmin = options.getfloat(section,"lmin")
		self.lmax = options.getfloat(section,"lmax")
		self.num_bins = options.getint(section,"num_bins")

		self.l_edges = np.logspace(np.log10(self.lmin),np.log10(self.lmax),self.num_bins+1)




