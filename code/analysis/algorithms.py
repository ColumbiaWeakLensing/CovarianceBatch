from lenstools.statistics.ensemble import Series,Ensemble
from lenstools.statistics.contours import ContourPlot

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
def fit_nbins(variance_ensemble,parameter="w",extra_columns=["bins"]):

	vmean = variance_ensemble

	#Compute variance expectation values
	vmean["1/nreal"] = vmean.eval("1.0/nreal")

	#Linear regression of the variance vs 1/nreal
	fit_results = dict()
	fit_results["nb"] = list()
	fit_results["s0"] = list()
	fit_results["nsim"] = list()

	for c in extra_columns:
		fit_results[c] = list()

	groupnsim = vmean.groupby("nsim")
	for g in groupnsim.groups:
		fit_results["nsim"].append(int(g))
		vmean_group = groupnsim.get_group(g)
		a,b,r_value,p_value,err = linregress(vmean_group["1/nreal"].values,vmean_group[parameter].values)
		fit_results["nb"].append(a/b)
		fit_results["s0"].append(b)

		for c in extra_columns:
			fit_results[c].append(vmean_group[c].mean())

	#Return to user
	return Ensemble.from_dict(fit_results)

#Fit all tables
def fit_nbins_all(db,parameter="w",extra_columns=["bins"]):

	nb_all = list()

	#Fit all tables
	for tbl in db.tables:
		v = db.read_table(tbl)
		nb = fit_nbins(v,parameter,extra_columns)
		nb["feature"] = tbl
		nb_all.append(nb)

	#Return to user
	return Ensemble.concat(nb_all,axis=0,ignore_index=True)

