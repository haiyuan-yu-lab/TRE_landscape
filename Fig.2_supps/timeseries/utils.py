# Download the code here: https://zenodo.org/records/5161189
# Adjust the code to fit our dataset requirment

import os
import logging
import math
import glob

import gzip
import numpy as np
import pandas as pd

from scipy.stats import zscore
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import multivariate_normal
from scipy.stats import chi2



def run_dpgp(mat_file, prefix, out_dir, tmp_dir, subsample=False, subsample_num=5000):
	"""Run DP-GP while managing files correctly
	"""
	run_shell_cmd("mkdir -p {}".format(out_dir))
	run_shell_cmd("mkdir -p {}".format(tmp_dir))
	
	# unzip file if needed
	input_mat_file = "{}/{}.dpgp.tmp".format(
		tmp_dir, os.path.basename(mat_file).split(".txt")[0])
	if mat_file.endswith(".gz"):
		unzip_mat = "zcat {0} > {1}".format(
			mat_file, input_mat_file)
		run_shell_cmd(unzip_mat)
	else:
		unzip_mat = "cat {0} > {1}".format(
			mat_file, input_mat_file)
		run_shell_cmd(unzip_mat)
	
	if subsample:
		subsampled_file = "{}.subsampled.tmp".format(input_mat_file.split(".tmp")[0])
		# keep header
		echo_header = "cat {0} | awk 'NR < 2' > {1}".format(
			input_mat_file, subsampled_file)
		os.system(echo_header)
		# subsample
		run_subsample = (
			"cat {0} | "
			"awk 'NR > 1' | "
			"shuf -n {1} --random-source={0} >> {2}").format(
			input_mat_file, subsample_num, subsampled_file)
		os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(run_subsample))
		os.system("rm {}".format(input_mat_file))
		input_mat_file = subsampled_file
		
	# TODO change header if need be?
	cluster = (
		"DP_GP_cluster.py -i {} -o {} --fast -p pdf --plot").format(
			input_mat_file,
			"{0}/{1}".format(out_dir, prefix))
	run_shell_cmd(cluster)
	
	# outfile name
	optimal_clusters = "{}/{}_optimal_clustering.txt".format(out_dir, prefix)
	
	return optimal_clusters



def filter_null_and_small_clusters(
        cluster_file,
        mat_file,
        out_cluster_file,
        ci=0.999, # sort of multiple hypothesis correction
        size_cutoff=0.2):
	"""Quick filters to remove trajectory groups whose multivariate Gaussian
	does NOT reject the null (ie, vector of zeros falls in the confidence
	interval) as well as small clusters
	
	Args:
	  mat_file: file with the timepoint data, with 0-column with ids
	  cluster_file: file with clusters, 0-column is clusters, 1-column is ids
	"""
	# first get sufficient statistics for the clusters
	cluster_means, cluster_covariances, cluster_sizes, cluster_names = get_cluster_sufficient_stats(
			cluster_file, mat_file)
	
	# set up size cutoff
	size_cutoff = size_cutoff * np.sum(cluster_sizes)
	
	# calculate what the chi2 cutoffs are based on the means and confidence intervals
	pdf_cutoff = chi2.pdf(1-ci, cluster_means[0].shape[0])
	indices_to_delete = []
	for cluster_idx in range(len(cluster_means)):
	
		# remove clusters by cluster size
		if cluster_sizes[cluster_idx] <= size_cutoff:
			indices_to_delete.append(cluster_idx)
			continue
	
		# calculate the multivariate normal (GP) and determine if cluster
		# does not reject the null.
		pdf_val = multivariate_normal.pdf(
			np.array([0 for i in range(cluster_means[0].shape[0])]),
			mean=cluster_means[cluster_idx],
			cov=cluster_covariances[cluster_idx],
			allow_singular=True)
		
		if pdf_val > pdf_cutoff:
			indices_to_delete.append(cluster_idx)
	
	cluster_names_to_delete = (np.array(indices_to_delete) + 1).tolist()
	
	# return cluster file
	cluster_list = pd.read_table(cluster_file)
	cluster_list = cluster_list[~cluster_list["cluster"].isin(cluster_names_to_delete)]
	
	# and now renumber
	clusters_remaining = cluster_list["cluster"].unique().tolist()
	renumbering = dict(zip(clusters_remaining, range(1, len(clusters_remaining)+1)))
	# print renumbering
	cluster_list["cluster"].replace(renumbering, inplace=True)
	cluster_list.columns = ["cluster", "id"]
	cluster_list = cluster_list.sort_values("cluster")
	
	# save out
	cluster_list.to_csv(out_cluster_file, sep="\t", index=False)
	
	return None



def run_shell_cmd(cmd): 
	"""Set up shell command runs
	"""
	logger = logging.getLogger(__name__)
	logger.debug(cmd)
	#subprocess.call('/bin/bash -c "$GREPDB"', shell=True)
	#os.system('GREPDB="{}"; /bin/bash -c "$GREPDB"'.format(cmd))
	os.system(cmd)
	
	if False:
	
		try:
			p = subprocess.Popen(cmd, shell=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				universal_newlines=True,
				preexec_fn=os.setsid)
			pid = p.pid
			pgid = os.getpgid(pid)
			ret = ''
			while True:
				line = p.stdout.readline()
				if line=='' and p.poll() is not None:
					break
				# log.debug('PID={}: {}'.format(pid,line.strip('\n')))
				print('PID={}: {}'.format(pid,line.strip('\n')))
				ret += line
			p.communicate() # wait here
			if p.returncode > 0:
				raise subprocess.CalledProcessError(
					p.returncode, cmd)
			return ret.strip('\n')
		except:
			# kill all child processes
			os.killpg(pgid, signal.SIGKILL)
			p.terminate()
			raise Exception('Unknown exception caught. PID={}'.format(pid))



def get_cluster_sufficient_stats(cluster_file, timeseries_file):
	"""From cluster file and timeseries file, get 
	means and covariances of the clusters and return as numpy array
	"""
	# open up cluster file
	clusters_df = pd.read_table(cluster_file, sep='\t')
	clusters_df.columns = ["cluster", "regions"]
	cluster_nums = list(set(clusters_df["cluster"].tolist()))
	
	# open up timeseries file and zscore
	timeseries_df = pd.read_table(timeseries_file, sep='\t', index_col=0)
	timeseries_df = timeseries_df.apply(zscore, axis=1, result_type="broadcast") #.to_frame()
	timeseries_df["regions"] = timeseries_df.index
	
	# merge the two
	# TODO make sure to remove examples that are not among the clusters
	merged_data = timeseries_df.merge(clusters_df, on="regions")
	del merged_data["regions"]
	
	# set up numpy out matrix
	cluster_means = []
	cluster_covariances = []
	cluster_sizes = []
	cluster_names = []
	
	# per cluster, get info
	for cluster_idx in range(len(cluster_nums)):
		cluster_num = cluster_idx + 1
	
		cluster_data = pd.DataFrame(merged_data[merged_data["cluster"] == cluster_num])
		del cluster_data["cluster"]
	
		# mean and covariance
		cluster_means.append(
			# cluster_data.mean(axis=0).as_matrix()
			cluster_data.mean(axis=0).to_numpy()
		)
		
		cluster_covariances.append(
			# cluster_data.cov().as_matrix()
			cluster_data.cov().to_numpy()
		) # note pandas already normalizes by N-1
	
		# other useful info
		cluster_sizes.append(cluster_data.shape[0])
		cluster_names.append(cluster_num)
		
	return cluster_means, cluster_covariances, cluster_sizes, cluster_names



def run_cluster_plotting_subworkflow(
		out_data,
		out_results,
		out_dir,
		prefix,
		mat_key,
		cluster_list_key):
	"""subsample and plot
	"""
	# assertions
	assert out_data.get(mat_key) is not None
	assert out_results.get(cluster_list_key) is not None
	
	run_shell_cmd("mkdir -p {}".format(out_dir))
	
	# subsample
	cluster_list_subsample_key = "{}.subsample.list".format(
		cluster_list_key.split(".list")[0])
	out_results[cluster_list_subsample_key] = "{}.subsampled.txt".format(
		out_results[cluster_list_key].split(".txt")[0])
	get_ordered_subsample(
		out_results[cluster_list_key],
		out_results[cluster_list_subsample_key])
	
	# plotting
	plot_dir = "{}/plots".format(out_dir)
	if not os.path.isdir(plot_dir):
		run_shell_cmd("mkdir -p {}".format(plot_dir))
	plot_clusters(
		out_results[cluster_list_key],
		out_results[cluster_list_subsample_key],
		out_data[mat_key],
		"{}/plots".format(out_dir),
		prefix,
		False)
	
	# splitting
	individual_cluster_files = glob.glob("{}/*cluster_*".format(out_dir))
	if len(individual_cluster_files) == 0:
		split_clusters(out_results[cluster_list_key])
	
	return out_data, out_results


def run_cluster_reordering_subworkflow(
        out_data,
        out_results,
        out_dir,
        prefix,
        mat_key,
        cluster_list_key):
	"""Reorder clusters based on hclust - try to order by time
	"""
	# assertions
	assert out_data.get(mat_key) is not None
	assert out_results.get(cluster_list_key) is not None
	
	run_shell_cmd("mkdir -p {}".format(out_dir))
	
	# run reordering
	cluster_list_reordered_key = "{}.reordered.list".format(
		cluster_list_key.split(".list")[0])
	out_results[cluster_list_reordered_key] = "{}/{}.reordered.clustering.txt".format(
		out_dir, os.path.basename(out_results[cluster_list_key]).split(".clustering")[0])
	reorder_clusters(
		out_results[cluster_list_key],
		out_data[mat_key],
		out_results[cluster_list_reordered_key])
	
	# run plotting subworkflow
	out_data, out_results = run_cluster_plotting_subworkflow(
		out_data,
		out_results,
		out_dir,
		prefix,
		mat_key,
		cluster_list_reordered_key)
	
	return out_data, out_results



def get_ordered_subsample(in_file, out_file, out_nrow=2000):
    """Given an input text file, grab an ordered sample
    """
    num_lines = 0
    with open(in_file, "r") as fp:
        for line in fp:
            num_lines += 1

    skip = math.ceil(float(num_lines) / out_nrow)

    num_lines = 0
    with open(out_file, "w") as out:
        with open(in_file, "r") as fp:
            for line in fp:
                if num_lines % skip == 0:
                    out.write(line)
                num_lines += 1
    
    return None



def plot_clusters(
        cluster_file,
        cluster_subsample_file,
        cluster_mat,
        out_dir,
        prefix,
        plot_individual=True):
	"""plots clusters given in the cluster file using the
	data in cluster mat
	"""
	# assertions
	assert os.path.isdir(out_dir)
	
	title = ""
	
	script_path = os.path.dirname(os.path.abspath(__file__))
	
	# heatmap plot
	r_plot_heatmap = (
		"{5}/viz.plot_timeseries_heatmap.R "
		"{0} {1} {2} {3} {4}").format(
			cluster_subsample_file, cluster_mat, out_dir, prefix, title, script_path)
	run_shell_cmd(r_plot_heatmap)
	
	
	return None



def reorder_clusters(cluster_file, cluster_mat, out_cluster_file):
	"""Sort clusters by hclust similarity (hierarchical clustering in sklearn)
	 """
	from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, to_tree
	#from scipy.spatial.distance import pdist, squareform
	from scipy.stats import zscore
	
	# first extract cluster means
	cluster_means, cluster_covariances, cluster_sizes, cluster_names = get_cluster_sufficient_stats(
		cluster_file, cluster_mat)
	
	means_z = zscore(np.array(cluster_means), axis=0)
	
	#cluster_dist = pdist(np.array(cluster_means), "euclidean")
	hclust = linkage(means_z, method="ward")
	
	# this is all the reordering code below
	if False:
		# using leaves_list and fcluster, determine split and reverse the FIRST half
		top_cut = fcluster(hclust, 2, criterion="maxclust")
		ordered_leaves = leaves_list(hclust)
		# for i in xrange(ordered_leaves.shape[0]):
		for i in range(ordered_leaves.shape[0]):
			current_leaf = ordered_leaves[i]
			if top_cut[current_leaf] == 2:
				# found the leaf
				split_point = i
				break
	
		# take left side of dendrogram and reverse
		# a recursive reordering of leaves by weight?
		ordered_leaves[0:split_point] = np.flip(ordered_leaves[0:split_point], axis=0)
		#ordered_leaves[split_point:] = np.flip(ordered_leaves[split_point:], axis=0)
		# print(ordered_leaves + 1)
	else:
		# try a recursive reordering
		hclust_tree = to_tree(hclust)
		old_ordered_leaves = leaves_list(hclust)

		try:
			reordered_tree = reorder_tree(hclust_tree, np.array(cluster_means))
			ordered_leaves = np.array(get_ordered_tree_nodes(reordered_tree))
		except Exception as e:
			ordered_leaves = old_ordered_leaves
	
		# print(old_ordered_leaves)
		# print(ordered_leaves)
		
	# build renumbering dict
	renumbering = dict(zip((ordered_leaves+1).tolist(), range(1, len(ordered_leaves)+1)))
	# print renumbering
	
	# read in cluster file
	cluster_list = pd.read_table(cluster_file)
	
	# renumber and sort
	cluster_list["cluster"].replace(renumbering, inplace=True)
	cluster_list = cluster_list.sort_values("cluster")
	cluster_list.to_csv(out_cluster_file, sep="\t", index=False)
	
	return None



def reorder_tree(tree, cluster_means):
	"""Recursive tool to reorder tree
	"""
	# go left
	if tree.left.count == 1:
		tree.left = tree.left
		left_nodes = [tree.left.id]
	else:
		# adjust the tree
		tree.left = reorder_tree(tree.left, cluster_means)
		left_nodes = get_ordered_tree_nodes(tree.left)
		
	# go right
	if tree.right.count == 1:
		tree.right = tree.right
		right_nodes = [tree.right.id]
	else:
		# adjust the tree
		tree.right = reorder_tree(tree.right, cluster_means)
		right_nodes = get_ordered_tree_nodes(tree.right)
		
	# calculate average cluster means for each set
	left_cluster_mean = np.sum(cluster_means[left_nodes,:], axis=0)
	right_cluster_mean = np.sum(cluster_means[right_nodes,:], axis=0)
	
	# extract the max
	left_max_idx = np.argmax(left_cluster_mean)
	right_max_idx = np.argmax(right_cluster_mean)
	
	# if max is at the edges, calculate slope to nearest 0
	flip = False
	if left_max_idx != right_max_idx:
		# good to go, carry on
		if left_max_idx > right_max_idx:
			flip = True
	else:
		# if left edge:
		if left_max_idx == 0:
			left_slope = left_cluster_mean[0] - left_cluster_mean[3]
			right_slope = right_cluster_mean[0] - right_cluster_mean[3]
			if left_slope < right_slope:
				flip = False
		# if right:
		elif left_max_idx == cluster_means.shape[1] - 1:
			left_slope = left_cluster_mean[-1] - left_cluster_mean[-3]
			right_slope = right_cluster_mean[-1] - right_cluster_mean[-3]
			if left_slope > right_slope:
				flip = True
		# if middle:
		else:
			left_side_max_idx = np.argmax(left_cluster_mean[[left_max_idx-1, left_max_idx+1]])
			right_side_max_idx = np.argmax(right_cluster_mean[[right_max_idx-1, right_max_idx+1]])
			if left_side_max_idx > right_side_max_idx:
				flip = True
	
	
	#import ipdb
	#ipdb.set_trace()
	
	# reorder accordingly
	if flip == True:
		right_tmp = tree.right
		left_tmp = tree.left
		tree.right = left_tmp
		tree.left = right_tmp
	
	return tree



def get_ordered_tree_nodes(tree):
	"""Recursively go through tree to collect nodes
	This will be in the order of leaveslist
	"""
	nodes = []
	# check left
	if tree.left.count == 1:
		nodes.append(tree.left.id)
	else:
		nodes += get_ordered_tree_nodes(tree.left)
		
	# check right
	if tree.right.count == 1:
		nodes.append(tree.right.id)
	else:
		nodes += get_ordered_tree_nodes(tree.right)
		
	# sum up and return
	return nodes



def split_clusters(cluster_file):
	"""Split cluster file into files of ids per cluster
	"""
	
	cluster_data = pd.read_table(cluster_file)
	cluster_data.columns = ["cluster", "id"]
	cluster_names = cluster_data["cluster"].unique().tolist()
	
	for cluster_name in cluster_names:
		# get that subset
		single_cluster = cluster_data.loc[cluster_data["cluster"] == cluster_name]
		
		# and write out
		out_file = "{}.cluster_{}.txt.gz".format(
			cluster_file.split(".clustering")[0],
			cluster_name)
		single_cluster.to_csv(
			out_file,
			columns=["id"],
			compression="gzip",
			sep='\t',
			header=False,
			index=False)
	
	return None



def assign_region_to_cluster(
        clusters_file,
        pooled_mat_file,
        out_hard_clusters_file,
        out_dir,
        prefix,
        ci=0.95,
        corr_cutoff=0.05):
	"""Given pooled timeseries files, go through
	regions and check for consistency after soft clustering
	"""
	run_shell_cmd("mkdir -p {0}/soft {0}/hard".format(out_dir))
	cluster_means, cluster_covariances, cluster_sizes, cluster_names = get_cluster_sufficient_stats(
		clusters_file, pooled_mat_file)
	
	# get pdf val for confidence interval
	# note that the multivariate normal pdf is distributed
	# as chi2(alpha) where alpha is the chosen significance
	pdf_cutoff = chi2.pdf(1-ci, cluster_means[0].shape[0])

	df = pd.read_table(pooled_mat_file, index_col=0)
	# It's Series in pandas 0.25.3
	df_zscores = df.apply(zscore, axis=1)

	with open(out_hard_clusters_file, "w") as out:
		out.write("cluster\tid\n")
		for region, timepoint_vector in df_zscores.items():
			hard_cluster = get_clusters_by_region(
				cluster_means,
				cluster_covariances,
				cluster_names,
				list(timepoint_vector),
				pdf_cutoff,
				corr_cutoff)
		
			if hard_cluster is not None:
				out.write("{}\t{}\n".format(hard_cluster, region))
		
	return None



def get_clusters_by_region(
        cluster_means,
        cluster_covariances,
        cluster_names,
        timepoint_vector,
        pdf_cutoff,
        corr_cutoff):
	"""Given a list of timepoint vectors, compare to all clusters
	utilizing a multivariate normal distribution
	"""

	soft_cluster_set = set()
	for cluster_idx in range(len(cluster_means)):

		# Determine if in confidence interval
		pdf_val = multivariate_normal.pdf(
			timepoint_vector,
			mean=cluster_means[cluster_idx],
			cov=cluster_covariances[cluster_idx],
			allow_singular=True) # this has to do with scaling? stack overflow 35273908

		if pdf_val < pdf_cutoff: # if not in confidence interval, continue
			continue
		soft_cluster_set.add(cluster_idx)		
	
	# Among the clusters, figure out best fit (to have a hard cluster assignment)
	best_pdf_val = 0
	hard_cluster = None
	for cluster_idx in soft_cluster_set:
		# get pdf val
		pdf_val = multivariate_normal.pdf(
			timepoint_vector,
			mean=cluster_means[cluster_idx],
			cov=cluster_covariances[cluster_idx],
			allow_singular=True)
		if pdf_val > best_pdf_val:
			best_pdf_val = pdf_val
			hard_cluster = cluster_names[cluster_idx]
	
	return hard_cluster






