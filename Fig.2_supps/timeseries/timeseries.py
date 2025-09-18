# Download the code here: https://zenodo.org/records/5161189
# Adjust the code to fit our dataset requirment

# Run the script in "DPGP" conda environment

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, help="Assay name")
parser.add_argument("--outdir", type=str)
parser.add_argument("--inputfile", type=str, help="Filepath to normalized data")
parser.add_argument("--size_cutoff", type=float, help="Cutoff for cluster size")

args = parser.parse_args()
results_dir, prefix, size_cutoff = args.outdir, args.prefix, args.size_cutoff
dpgp_pooled_dir = "{}/pooled".format(results_dir)
out_data = {}
out_data[prefix] = args.inputfile
out_results = {}



#------------------------------------------------------------------------------------------------------------------
# STEP 1: Get the initial set of time series clusters
# The signal matrix with the pooled data is subsampled (n=5000 for speed, since the algorithm was originally built to run effectively at the scale of thousands of genes, not tens of thousands of regions) with the default parameters, providing the initial set of time series clusters.

utils.run_dpgp(
	out_data[prefix],
	"{}.pooled".format(prefix), 
	dpgp_pooled_dir, 
	results_dir,
	subsample=True)



#------------------------------------------------------------------------------------------------------------------
# STEP 2: Plot heatmaps of subsampled data

# Elements are grouped by clusters; values refer to logFC relative to the earliest timepoint
clusters_raw_handle = "clusters.raw.list"
out_results[clusters_raw_handle] = "{0}/{1}.pooled_optimal_clustering.txt".format(dpgp_pooled_dir, prefix)

out_data, out_results = utils.run_cluster_plotting_subworkflow(
	out_data,
	out_results,
	dpgp_pooled_dir,
	"{}.pooled".format(prefix),
	prefix,
	clusters_raw_handle)

# Reorder clusters based on hclust - try to order by time
reorder_dir = "{}/reordered".format(dpgp_pooled_dir)
out_data, out_results = utils.run_cluster_reordering_subworkflow(
	out_data,
	out_results,
	reorder_dir,
	"{}.raw.reordered".format(prefix),
	prefix,
	clusters_raw_handle)

clusters_raw_reordered_handle = "{}.reordered.list".format(
	clusters_raw_handle.split(".list")[0])



#------------------------------------------------------------------------------------------------------------------
# STEP 3: Cluster filtering
# 1. The cluster set is filtered for cluster size such that any cluster that has a total membership of elements < XXX% of all dynamically transcribed elements is removed. 
# 2. The cluster set is further filtered to remove non-dynamic trajectories, which are the clusters whose multivariate Gaussian process does not reject the null hypothesis of no change across time.

nullfilt_dir = "{}/nullfilt".format(results_dir)
utils.run_shell_cmd("mkdir -p {}".format(nullfilt_dir))
clusters_nullfilt_handle = "clusters.null_filt.list"
out_results[clusters_nullfilt_handle] = "{0}/{1}.nullfilt.clustering.txt".format(nullfilt_dir, prefix)

utils.filter_null_and_small_clusters(
	out_results[clusters_raw_reordered_handle],
	out_data[prefix],
	out_results[clusters_nullfilt_handle],
	ci=0.95,
	size_cutoff=size_cutoff)

# Reorder clusters based on hclust
reorder_dir = "{}/reordered".format(nullfilt_dir)
out_data, out_results = utils.run_cluster_reordering_subworkflow(
	out_data,
	out_results,
	reorder_dir,
	"{}.nullfilt.reordered".format(prefix),
	prefix,
	clusters_nullfilt_handle)



#------------------------------------------------------------------------------------------------------------------
# STEP 4: Assign each element to a cluster
# For each element, with its corresponding signal trajectory across time, we assign the element to each cluster that it could match. The element matches a cluster if it’s in the multivariate confidence interval (CI 0.95) for the trajectory. If there is more than one matched cluster, the element is assigned to the best fit. If there are no matched cluster, the element is discarded.

full_dir = "{}/full".format(results_dir)
clusters_full_hard_handle = "clusters.full.hard.list"
out_results[clusters_full_hard_handle] = ("{0}/hard/{1}.full.hard.clustering.txt").format(full_dir, prefix)

utils.assign_region_to_cluster(
	out_results[clusters_nullfilt_handle],
	out_data[prefix],
	out_results[clusters_full_hard_handle],
	full_dir,
	"{}.full".format(prefix),
	ci=0.95,
	corr_cutoff=0.05)

# Reorder clusters based on hclust
reorder_dir = "{}/hard/reordered".format(full_dir)
out_data, out_results = utils.run_cluster_reordering_subworkflow(
	out_data,
	out_results,
	reorder_dir,
	"{}.full.hard.reordered".format(prefix),
	prefix,
	clusters_full_hard_handle)



