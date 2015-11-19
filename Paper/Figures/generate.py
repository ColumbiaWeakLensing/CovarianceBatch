#!/usr/bin/env python

import sys,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.database import Database

import variance_scaling

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

#Power spectrum variance
def ps_variance(cmd_args,nsim=[1,2,5,10,50,100],colors=["black","blue","green","red","purple","orange"],fontsize=18):

	assert len(colors)>=len(nsim)

	#Multipoles and number of modes
	ell = np.load("features/ell.npy")
	num_ell = np.load("features/num_ell.npy")

	#Plot
	fig,ax = plt.subplots()
	for nc,ns in enumerate(nsim):

		#Load the relevant ensemble,compute mean and variance
		ens = Ensemble(np.load("features/Maps{0}/power_spectrum_s0.npy".format(ns))).head(1000)
		p = ens.mean().values
		pcov = ens.cov().values.diagonal()

		#Scale the covariance to the number of modes
		pcov = num_ell*pcov/(p**2)

		#Plot
		ax.plot(ell,pcov,color=colors[nc],label=r"$N_s={0}$".format(ns))

	#Labels
	ax.set_xlabel(r"$l$",fontsize=fontsize)
	ax.set_ylabel(r"$\mathrm{Var}(P^{\kappa\kappa}_l)\times N_l/P^{\kappa\kappa,2}_l$",fontsize=fontsize)
	ax.legend(loc="upper left")

	#Save
	fig.savefig("ps_variance."+cmd_args.type)

#Power spectrum pdf
def ps_pdf(cmd_args,nell=[0,4,9,14],nsim=[1,2,5,50,100],colors=["black","blue","green","red","purple"],fontsize=18):

	assert len(colors)>=len(nsim)
	assert len(nell)==4

	#Multipoles and number of modes
	ell = np.load("features/ell.npy")

	#Plot
	fig,ax = plt.subplots(2,2,figsize=(16,12))
	for nc,ns in enumerate(nsim):

		#Load the relevant ensemble,compute mean and variance
		ens = Ensemble(np.load("features/Maps{0}/power_spectrum_s0.npy".format(ns))).head(1000)
		
		#Fill each sub plot
		for na,subax in enumerate(ax.reshape(4)):
			subax.hist(ens[nell[na]].values,histtype="step",bins=50,normed=True,label=r"$N_s={0}$".format(ns),color=colors[nc])

	#Plot the result for the BIG ensemble generated with 1 simulation and 128000 realizations
	ens = Ensemble(np.load("features/MillionMapsPower/power_spectrum_s0.npy"))

	#Fill each sub plot
	for na,subax in enumerate(ax.reshape(4)):
		subax.hist(ens[nell[na]].values,histtype="step",bins=50,normed=True,color=colors[0],linewidth=3)

	#Labels
	for na,subax in enumerate(ax.reshape(4)):
		subax.set_xlabel(r"$P^{\kappa\kappa}_l$",fontsize=fontsize)
		subax.set_ylabel(r"$\mathcal{L}(P^{\kappa\kappa}_l)$",fontsize=fontsize)
		subax.set_title(r"$l={0}$".format(int(ell[nell[na]])),fontsize=fontsize)
		subax.legend()

	#Save
	fig.tight_layout()
	fig.savefig("ps_pdf."+cmd_args.type)

#Scaling of the variance with Nr
def scaling_nr(cmd_args,db_filename="variance_scaling_power_expected.sqlite",parameter="w",nsim=[1,2,5,50,100],colors=["black","blue","green","red","purple"],fontsize=18):

	assert len(colors)>=len(nsim)

	#Plot panel
	fig,ax = plt.subplots()

	####################################################################################################################################

	#Open the database and look for different nsim
	with Database("data/"+db_filename) as db:
		v = db.query("SELECT nsim,nreal,{0} FROM variance_all WHERE nsim IN ({1})".format(parameter,",".join([str(n) for n in nsim])))

	#Fit with the Dodelson scaling and overlay the fit
	vfit = variance_scaling.fit_nbins(v,parameter=parameter)
	v = Ensemble.merge(v,vfit,on="nsim")
	v[parameter+"_fit"] = v.eval("s0*nb/nreal")
	v[parameter+"_subtracted"] = v.eval("{0}-s0".format(parameter))
	nsim_group = v.groupby("nsim")

	for nc,ns in enumerate(nsim):
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_subtracted",linestyle="-",color=colors[nc],ax=ax,label=r"$N_s={0}$".format(ns))
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_fit",linestyle="--",color=colors[nc],ax=ax,legend=False)

	####################################################################################################################################
	###############Same thing for the BIG simulation set################################################################################
	####################################################################################################################################

	#Open the database and look for different nsim
	with Database("data/variance_scaling_power_largeNr.sqlite") as db:
		v = db.query("SELECT nsim,nreal,{0} FROM variance_all".format(parameter))

	#Fit with the Dodelson scaling and overlay the fit
	vfit = variance_scaling.fit_nbins(v,parameter=parameter)
	v = Ensemble.merge(v,vfit,on="nsim")
	v[parameter+"_fit"] = v.eval("s0*nb/nreal")
	v[parameter+"_subtracted"] = v.eval("{0}-s0".format(parameter))
	nsim_group = v.groupby("nsim")

	for ns in [1]:
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_subtracted",linestyle="-",linewidth=2,color=colors[0],ax=ax,legend=False)
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_fit",linestyle="--",linewidth=2,color=colors[0],ax=ax,legend=False)

	####################################################################################################################################

	#Axes scale
	ax.set_xscale("log")
	ax.set_yscale("log")

	#Labels
	ax.set_xlabel(r"$N_r$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle\hat{\sigma}^2_w\rangle$",fontsize=fontsize)

	#Save
	fig.savefig("scaling_nr."+cmd_args.type)


#Scaling of the variance with Ns
def scaling_ns(cmd_args,db_filenames=["variance_scaling_expected.sqlite","variance_scaling_expected_diagonal.sqlite"],parameter="w",nreal=range(100,1000,200),colors=["black","blue","green","red","purple"],linestyles=["-","--"],fontsize=18):

	assert len(colors)>=len(nreal)

	#Plot panel
	fig,ax = plt.subplots()

	for ndb,db_filename in enumerate(db_filenames):

		if ndb>0:
			legend=False
		else:
			legend = True
	
		#Open the database and look for different nsim
		with Database("data/"+db_filename) as db:
			v = db.read_table("variance")

		#Fit with the Dodelson scaling
		vfit = variance_scaling.fit_nbins(v,parameter=parameter)
		v = Ensemble.merge(v,vfit,on="nsim")
		v[parameter+"_ns"] = v.eval(parameter+"/(1+(nb/nreal))")
		nreal_group = v.groupby("nreal")

		for nc,nr in enumerate(nreal):
			nreal_group.get_group(nr).plot(x="nsim",y=parameter+"_ns",color=colors[nc],ax=ax,label=r"$N_r={0}$".format(nr),linestyle=linestyles[ndb],legend=legend)

	#Labels
	ax.set_xlabel(r"$N_s$",fontsize=fontsize)
	ax.set_ylabel(r"$\sigma_0^2(N_s)$",fontsize=fontsize)

	#Save
	fig.savefig("scaling_ns."+cmd_args.type)

		
###########################################################################################################################################

#Method dictionary
method = dict()
method["1"] = ps_pdf
method["2"] = ps_variance
method["3"] = scaling_nr
method["4"] = scaling_ns

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()