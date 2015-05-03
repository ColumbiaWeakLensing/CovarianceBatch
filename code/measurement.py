import sys,os

from lenstools import Ensemble

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

available = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]

def suffix(n):
	if n==1:
		return ""
	else:
		return str(n)


###################################################################
#####################Class handler for descriptors#################
###################################################################

class Measurement(object):

	def __init__(self,descriptor,smoothing_scale=1.0*u.arcmin):

		self.descriptor = descriptor
		self.smoothing_scale = smoothing_scale
		self.available = available

	def call(self,callback,shape=None,**kwargs):

		if shape is None:
			self.data = np.zeros(len(self.available))
		else:
			self.data = np.zeros((len(self.available),) + shape)
		
		for i,n in enumerate(self.available):
			ensemble_filename = os.path.join("..","Om0.260_Ol0.740_w-1.000_ns0.960_si0.800","512b240","Maps"+suffix(n),self.descriptor+"_s{0}.npy".format(int(self.smoothing_scale.to(u.arcmin).value)))
			ens = Ensemble.read(ensemble_filename)
			self.data[i] = callback(ens,**kwargs)

	def plot(self,fig=None,ax=None,**kwargs):

		if (fig is None) or (ax is None):
			self.fig,self.ax = plt.subplots()
		else:
			self.fig = fig
			self.ax = ax

		self.ax.plot(self.available,self.data,**kwargs)
		self.ax.legend()


###################################################################
#####################Recurrent callbacks###########################
###################################################################

def dP_over_P(ens,ell,num_ell,bin_number,**kwargs):

	p = ens.mean()
	dP = np.sqrt(ens.covariance(**kwargs).diagonal())

	return ((dP/p)[bin_number]**2) * num_ell[bin_number]

def fullCovariance(ens):
	return ens.covariance()
