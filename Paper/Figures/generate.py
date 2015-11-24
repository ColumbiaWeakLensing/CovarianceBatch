#!/usr/bin/env python

import sys,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.database import Database

import algorithms

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

#Power spectrum variance
def ps_variance(cmd_args,nsim=[1,2,5,10,50,100],colors=["black","blue","green","red","purple","orange"],fontsize=22):

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
def ps_pdf(cmd_args,nell=[0,4,9,14],nsim=[1,2,5,50,100],colors=["black","blue","green","red","purple"],fontsize=22):

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
def scaling_nr(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",parameter="w",nsim=[1,2,5,50,100],colors=["black","blue","green","red","purple"],fontsize=22):

	assert len(colors)>=len(nsim)

	#Plot panel
	fig,ax = plt.subplots()

	####################################################################################################################################

	#Open the database and look for different nsim
	with Database("data/"+db_filename) as db:
		v = db.query("SELECT nsim,nreal,bins,{0} FROM power_logb_all WHERE nsim IN ({1})".format(parameter,",".join([str(n) for n in nsim])))

	#Fit with the Dodelson scaling and overlay the fit
	vfit = algorithms.fit_nbins(v,parameter=parameter)
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
		v = db.query("SELECT nsim,nreal,bins,{0} FROM power_logb_all".format(parameter))

	#Fit with the Dodelson scaling and overlay the fit
	vfit = algorithms.fit_nbins(v,parameter=parameter)
	v = Ensemble.merge(v,vfit,on="nsim")
	v[parameter+"_fit"] = v.eval("s0*nb/nreal")
	v[parameter+"_subtracted"] = v.eval("{0}-s0".format(parameter))
	nsim_group = v.groupby("nsim")

	for ns in [1]:
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_subtracted",linestyle="-",linewidth=1.5,color=colors[0],ax=ax,legend=False)
		nsim_group.get_group(ns).plot(x="nreal",y=parameter+"_fit",linestyle="--",linewidth=1.5,color=colors[0],ax=ax,legend=False)

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
def scaling_ns(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",features=["power_logb_all","power_all","peaks_all"],colors=["black","red","green"],parameter="w",fontsize=22):

	assert len(colors)==len(features)

	#Plot panel
	fig,ax = plt.subplots()

	#Labels
	labels = {
	"power_logb_large" : None,
	"power_logb_small" : None,
	"power_logb_all" : "Power spectrum log binning",
	"power_large" : None,
	"power_small" : None,
	"power_large+small" : None,
	"power_all" : "Power spectrum linear binning",
	"peaks_low" : None,
	"peaks_intermediate" : None,
	"peaks_high" : None,
	"peaks_low+intermediate" : None,
	"peaks_intermediate+high" : None,
	"peaks_all" : "Peak counts",
	} 

	#Load the database and fit for the effective dimensionality of each feature space
	with Database("data/"+db_filename) as db:
		nb_fit = algorithms.fit_nbins_all(db,parameter)

	#Plot the variance coefficient for each feature
	for nc,f in enumerate(features):
		nb_fit_feature = nb_fit.query("feature=='{0}'".format(f)).sort_values("nsim")
		nb_fit_feature["relative"] = nb_fit_feature["s0"] / nb_fit_feature["s0"].mean() 
		nb_fit_feature.plot(x="nsim",y="relative",ax=ax,color=colors[nc],label=labels[f],legend=False)

	#Labels
	ax.set_xlim(-10,210)
	ax.set_xlabel(r"$N_s$",fontsize=fontsize)
	ax.set_ylabel(r"$\sigma_0^2(N_s)/\sigma_{0,mean}^2$",fontsize=fontsize)
	ax.legend()

	#Save
	fig.savefig("scaling_ns."+cmd_args.type)

#Scaling of the effective dimensionality with Nb
def effective_nb(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",parameter="w",correct=False,evaluate=lambda nb:nb-3,th_label=r"$N_b-N_p$",fontsize=22,figname="effective_nb"):

	#Plot panel
	fig,ax = plt.subplots()

	##############################
	#Plot styles for each feature#
	##############################

	#Markers
	markers = {
	"power_logb_large" : "x",
	"power_logb_small" : "x",
	"power_logb_all" : "x",
	"power_large" : "+",
	"power_small" : "x",
	"power_large+small" : "o",
	"power_all" : "o",
	"peaks_low" : "+",
	"peaks_intermediate" : "*",
	"peaks_high" : "d",
	"peaks_low+intermediate" : "x",
	"peaks_intermediate+high" : "o",
	"peaks_all" : "s",
	} 

	#Colors
	colors = {
	"power_logb_large" : "black",
	"power_logb_small" : "black",
	"power_logb_all" : "black",
	"power_large" : "red",
	"power_small" : "red",
	"power_large+small" : "red",
	"power_all" : "red",
	"peaks_low" : "green",
	"peaks_intermediate" : "green",
	"peaks_high" : "green",
	"peaks_low+intermediate" : "green",
	"peaks_intermediate+high" : "green",
	"peaks_all" : "green",
	} 

	#Labels
	labels = {
	"power_logb_large" : "Power spectrum log binning",
	"power_logb_small" : None,
	"power_logb_all" : None,
	"power_large" : None,
	"power_small" : None,
	"power_large+small" : None,
	"power_all" : "Power spectrum linear binning",
	"peaks_low" : "Peak counts",
	"peaks_intermediate" : None,
	"peaks_high" : None,
	"peaks_low+intermediate" : None,
	"peaks_intermediate+high" : None,
	"peaks_all" : None,
	} 

	#Offsets
	offsets = {
	"power_logb_large" : 0,
	"power_logb_small" : 0,
	"power_logb_all" : 0,
	"power_large" : -0.5,
	"power_small" : -1,
	"power_large+small" : 0,
	"power_all" : 0,
	"peaks_low" : 0.5,
	"peaks_intermediate" : 1,
	"peaks_high" : 1.5,
	"peaks_low+intermediate" : 0.5,
	"peaks_intermediate+high" : 1,
	"peaks_all" : 0,
	} 
	
	#################################################################################################################

	#Load the database and fit for the effective dimensionality of each feature space
	with Database("data/"+db_filename) as db:
		features = db.tables
		nb_fit = algorithms.fit_nbins_all(db,parameter,correct=correct)

	#Scatter each feature
	for f in features:
		nb_fit_feature = nb_fit.query("feature=='{0}'".format(f))
		ax.scatter(nb_fit_feature["bins"]+offsets[f],nb_fit_feature["nb"],color=colors[f],marker=markers[f],s=10+(100-10)*(nb_fit_feature["nsim"]-1)/(200-1),label=labels[f])

	#Plot the theory prediction
	bins = np.arange(1,51)
	ax.plot(bins,evaluate(bins),color="black",linewidth=2,label=th_label)

	#Axis labels
	ax.set_xlim(0,53)
	ax.set_xlabel(r"$N_b$",fontsize=fontsize)
	ax.set_ylabel(r"$D$",fontsize=fontsize)
	ax.legend(loc="upper left",prop={"size":15})

	#Save the figure
	fig.savefig(".".join([figname,cmd_args.type]))


def effective_nb_gaussian(cmd_args):
	effective_nb(cmd_args,db_filename="variance_scaling_gaussian_expected.sqlite",parameter="w",correct=False,evaluate=lambda nb:nb-3,th_label=r"$N_b-N_p$",fontsize=22,figname="effective_nb_gaussian")

		
###########################################################################################################################################

#Method dictionary
method = dict()
method["1"] = ps_pdf
method["2"] = ps_variance
method["3"] = scaling_nr
method["4"] = scaling_ns
method["5a"] = effective_nb
method["5b"] = effective_nb_gaussian

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()