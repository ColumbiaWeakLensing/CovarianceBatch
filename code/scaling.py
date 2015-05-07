import numpy as np

def variance_scaling(ens,group_sizes,num_samples=100):

	variance = []

	for group_size in group_sizes:
		ens_grouped = ens.group(group_size,"sparse",inplace=False).subset(range(num_samples))
		variance.append(ens_grouped.covariance().diagonal())

	return np.array(variance)
	