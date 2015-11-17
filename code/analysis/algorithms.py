from lenstools.statistics.ensemble import Series,Ensemble
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

##########################################
###Fit for the effective number of bins###
##########################################

def fit_nbins(variance_ensemble,parameter="w"):

	vmean = variance_ensemble

	#Compute variance expectation values
	vmean["1/nreal"] = vmean.eval("1.0/nreal")

	#Linear regression of the variance vs 1/nreal
	fit_results = dict()
	fit_results["nb"] = list()
	fit_results["s0"] = list()
	fit_results["nsim"] = list()

	groupnsim = vmean.groupby("nsim")
	for g in groupnsim.groups:
		fit_results["nsim"].append(int(g))
		vmean_group = groupnsim.get_group(g)
		a,b,r_value,p_value,err = linregress(vmean_group["1/nreal"].values,vmean_group[parameter].values)
		fit_results["nb"].append(a/b)
		fit_results["s0"].append(b)

	#Return to user
	return Ensemble.from_dict(fit_results)