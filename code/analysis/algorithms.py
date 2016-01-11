from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.contours import ContourPlot

import numpy as np
from scipy.stats import linregress

###############################
###Bootstrap fisher ellipses###
###############################

def bootstrap_fisher(ensemble,fisher,true_covariance,extra_items):
	
	pcov = fisher.parameter_covariance(ensemble.cov(),observed_features_covariance=true_covariance)
	pvar = Series(pcov.values.diagonal(),index=pcov.index)
	for key in extra_items.keys():
		pvar[key] = extra_items[key]
	
	return pvar.to_frame().T

def bootstrap_fisher_diagonal(ensemble,fisher,true_covariance,extra_items):

	cov = Ensemble(np.diag(ensemble.var().values),index=true_covariance.index,columns=true_covariance.columns)
	pcov = fisher.parameter_covariance(cov,observed_features_covariance=true_covariance)
	pvar = Series(pcov.values.diagonal(),index=pcov.index)
	for key in extra_items.keys():
		pvar[key] = extra_items[key]
	
	return pvar.to_frame().T

###############################
###Bootstrap full likelihood###
###############################

def bootstrap_area(ensemble,emulator,parameter_grid,fisher,true_covariance,test_data,extra_items,grid_mpi_pool):

	scores = emulator.score(parameter_grid,test_data,features_covariance=ensemble.cov(),pool=grid_mpi_pool)
	scores["likelihood"] = scores.eval("exp(-0.5*{0})".format(emulator.feature_names[0]))
	contour = ContourPlot.from_scores(scores,parameters=["Om","sigma8"],feature_names="likelihood")
	contour.getLikelihoodValues([0.684],precision=0.01)
	area = contour.confidenceArea()

	parea = Series([area.keys()[0],area.values()[0]],index=["p_value","area"])
	for key in extra_items:
		parea[key] = extra_items[key]

	return parea.to_frame().T

##########################################
###Fit for the effective number of bins###
##########################################

#Fit a single table
def fit_nbins(variance_ensemble,parameter="w",extra_columns=["bins"],correct=False,kind="quadratic",vfilter=None,nreal_min=500):

	if vfilter is not None:
		vmean = vfilter(variance_ensemble).copy()
	else:
		vmean = variance_ensemble
	
	if correct:
		assert "bins" in extra_columns
		vmean[parameter] = vmean.eval("{0}*(nreal-1)/(nreal-bins-2)".format(parameter))

	#Compute variance expectation values
	vmean["1/nreal"] = vmean.eval("1.0/nreal")

	#Linear regression of the variance vs 1/nreal
	fit_results = dict()
	fit_results["D"] = list()
	fit_results["s0"] = list()
	fit_results["nsim"] = list()

	if kind=="quadratic":
		fit_results["D2"] = list()
		fit_results["res_exponent"] = list()

	if kind=="subtract":
		fit_results["decay_exponent"] = list()

	for c in extra_columns:
		fit_results[c] = list()

	groupnsim = vmean.groupby("nsim")
	for g in groupnsim.groups:
		fit_results["nsim"].append(int(g))
		vmean_group = groupnsim.get_group(g)

		if kind=="linear":
			
			#Perform a simple linear regression
			a,b,r_value,p_value,err = linregress(vmean_group["1/nreal"],vmean_group[parameter])
			fit_results["D"].append(a/b)
			fit_results["s0"].append(b)

		elif kind=="subtract":

			#Let the variance with the highest nreal be the intercept
			vmean_group_sorted = vmean_group.sort_values("1/nreal")
			s0 = vmean_group_sorted[parameter].iloc[0]
			fit_results["s0"].append(s0)
			a,b,r_value,p_value,err = linregress(np.log(vmean_group_sorted["nreal"].values[1:]),np.log(np.abs((vmean_group_sorted[parameter]-s0).values[1:])))

			fit_results["decay_exponent"].append(a)
			fit_results["D"].append(np.exp(b)/s0)

		elif kind=="quadratic":

			#Perform a linear regression on the high Nr tail
			vmean_group_high_nr = vmean_group.query("nreal>={0}".format(nreal_min)).sort_values("1/nreal")
			a,b,r_value,p_value,err = linregress(vmean_group_high_nr["1/nreal"],vmean_group_high_nr[parameter])
			fit_results["D"].append(a/b)
			fit_results["s0"].append(b)

			#Fit the residuals with a quadratic
			residuals = vmean_group[parameter] - a*vmean_group["1/nreal"] - b
			rexp,logd2,r_value,p_value,err = linregress(np.log(vmean_group["1/nreal"]),np.log(np.abs(residuals)))
			fit_results["res_exponent"].append(rexp)
			fit_results["D2"].append(np.exp(logd2)/b)


		else:
			raise NotImplementedError("Fit of kind '{0}' not implemented yet!".format(kind))


		for c in extra_columns:
			fit_results[c].append(vmean_group[c].mean())

	#Return to user
	return Ensemble.from_dict(fit_results)

#Fit all tables
def fit_nbins_all(db,**kwargs):

	nb_all = list()

	#Fit all tables
	for tbl in db.tables:
		v = db.read_table(tbl)
		nb = fit_nbins(v,**kwargs)
		nb["feature"] = tbl
		nb_all.append(nb)

	#Return to user
	return Ensemble.concat(nb_all,axis=0,ignore_index=True)

#Fit a single table, but vary the threshold for the maximum Nr
def fit_nbins_nr(variance_ensemble,nrmax,**kwargs):

	nb_all = list()

	#Fit all Nr tables
	for nr in nrmax:
		nb = fit_nbins(variance_ensemble,vfilter=lambda db:db.query("nreal<={0}".format(nr)),**kwargs)
		nb["nreal_max"] = nr
		nb_all.append(nb)

	#Return to user
	return Ensemble.concat(nb_all,axis=0,ignore_index=True)



