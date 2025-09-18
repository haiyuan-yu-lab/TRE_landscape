def get_count_matrix(inputfile, outputfile, samples, bws):
	"""
	Get total read counts for each element across samples.
	"""

	import pandas as pd
	import pybedtools
	import pyBigWig
	
	merged = pybedtools.BedTool(inputfile)
	columns = ["_".join(i.fields[:3]) for i in merged]
	
	results = []
	for s in samples:
		pl, mn = bws[s]
		bw1 = pyBigWig.open(pl)
		bw2 = pyBigWig.open(mn)
		row = []
		for c in columns:
			# PINTS: 0-based start, 0-based end; both are included 
			chrom, start, end = c.split("_")
			start = int(start)
			end = int(end)
			reads1 = 0
			reads2 = 0
			# pyBigWig: 0-based start, 0-based end; the end is not included
			if bw1.intervals(chrom, start, end+1) != None:
				for start2, end2, score in bw1.intervals(chrom, start, end+1):
					l = min(end+1, end2) - max(start, start2)
					reads1 += abs(score) * l
			if bw2.intervals(chrom, start, end+1) != None:
				for start2, end2, score in bw2.intervals(chrom, start, end+1):
					l = min(end+1, end2) - max(start, start2)
					reads2 += abs(score) * l
			# Total read counts
			row.append(reads1+reads2)

		results.append(row)
	df = pd.DataFrame(results, columns=columns, index=samples)
	df = df.T
	df.to_csv(outputfile, sep="\t")



def age_dissection(inputfile, clades):	
	"""Assign age to each sequence block"""
	from biodata.delimited import DelimitedReader
	import numpy as np
	import pandas as pd
	from collections import defaultdict
	
	age_per_block = defaultdict(dict)
	mapping = {}
	with DelimitedReader(inputfile) as dr:
		for cols in dr:
			chrom, pstart, pend, pstart2, pend2 = cols[:5]
			mapping[(chrom, pstart, pend)] = (chrom, pstart2, pend2)
			bstart, bend, clade, overlap = cols[-4:]
			# Only include regions with overlapping length >= 6bp
			if int(overlap) >= 6:
				block_start = np.max([int(pstart), int(bstart)])
				block_end = np.min([int(pend), int(bend)])
				# Use original boundary
				age_per_block[(chrom, pstart, pend)][(block_start, block_end)] = clades[clade]

	results = []
	index = []
	for r in age_per_block:
		chrom, start, end = r
		index.append("_".join(mapping[r]))
		bp_features = []
		for i in range(int(start), int(end)):
			feature = np.nan
			for block in age_per_block[r]:
				b_s, b_e = block
				if b_s <= i <= b_e:
					feature = age_per_block[r][block]
					break
			bp_features.append(feature)
		results.append(bp_features)

	df = pd.DataFrame(results, index=index)
	return df



def get_pause_distances(inputfile):
	"""Calculate pause distance for a given sample"""
	from biodata.bed import BEDGraphReader
	from collections import Counter
	
	cter = Counter()
	with BEDGraphReader(inputfile, dataValueType=lambda s: Counter(map(int, s.split(",")))) as br:
		for b in br:
			for k, v in b.dataValue.items():
				cter[abs(k)] += v
	return cter



#--------------------------------------------------------------------------------------------------------
# A bunch of functions for plotting...
#--------------------------------------------------------------------------------------------------------

def generate_feature_metaplot(df, palette, hue_order, ax, test=True, errorbar=('ci', 95)):
	"""
	Generate a metaplot for a given feature. 
	"""

	import pandas as pd
	import seaborn as sns
	
	if test:
		frames = []
		for n in range(len(hue_order)):
			frames.append(df[df["Label"]==hue_order[n]].head(10))
		df = pd.concat(frames)

	# lineplot drops NAs from the DataFrame before plotting
	sns.lineplot(data=df, x="Position", y="Feature", hue="Label", hue_order=hue_order, palette=palette, ax=ax, errorbar=errorbar)



def get_reads(region, bws, samples_dict):
	"""Get a dataframe of average RPM values for plotting browser shot"""
	from collections import defaultdict
	import pyBigWig
	import pandas as pd
	
	chrom, start, end = region.split("_")
	start = int(start)
	end = int(end)
	
	reads1 = {}
	reads2 = {}
	for g in samples_dict:
		reads1[g] = defaultdict(int)
		reads2[g] = defaultdict(int)
		for s in samples_dict[g]:
			pl, mn = bws[s]
			bw1 = pyBigWig.open(pl)
			bw2 = pyBigWig.open(mn)
			if bw1.intervals(chrom, start, end+1) != None:
				for start2, end2, score in bw1.intervals(chrom, start, end+1):
					for i in range(start2, end2+1):
						reads1[g][i] += score
			if bw2.intervals(chrom, start, end+1) != None:
				for start2, end2, score in bw2.intervals(chrom, start, end+1):
					for i in range(start2, end2+1):
						reads2[g][i] += score
	# Average RPM
	results = []
	for g in samples_dict:
		for i in range(start, end+1):
			results.append([g, i, reads1[g][i]/len(samples_dict[g]), "fwd"])
			results.append([g, i, reads2[g][i]/len(samples_dict[g]), "rev"])
	df = pd.DataFrame(results, columns=["type", "position", "rpm", "orientation"])

	return df



def gene_representation(gene, transcript, gtf, ax, zoom_start, zoom_end, fontsize, color="#000000"):
	"""A schematic visualization of a gene’s structure, including exons, introns, and strand"""
	import matplotlib.patches as patches
	
	exons = gtf[
	    (gtf["feature"] == "exon") &
	    (gtf["attribute"].str.contains(transcript))
		]
	chrom = exons["chrom"].iloc[0]
	strand = exons["strand"].iloc[0]
	gene_start = exons["start"].min()
	gene_end = exons["end"].max()
	
	# Intron line
	ax.hlines(y=0.5, xmin=gene_start, xmax=gene_end, color="black", linewidth=1)
	
	# Exon rectangles
	for _, row in exons.iterrows():
	    ax.add_patch(patches.Rectangle((row["start"], 0), row["end"] - row["start"], 1, color="black"))
	
	# Arrows indicating strand
	step = max((gene_end - gene_start) // 20, 1000)
	arrow_y = 0.5
	for x in range(gene_start + step, gene_end, step):
	    dx = step if strand == "+" else -step
	    ax.arrow(x, arrow_y, dx, 0, head_width=0.5, head_length=step // 3,
	             facecolor="black", edgecolor=None, length_includes_head=True)

	ax.set_title(gene, fontsize=fontsize, c=color, y=1.1)

	# Highlight the region for zoom-in view 
	rect = patches.Rectangle(
	    (zoom_start, 0),               
	    zoom_end - zoom_start,         
	    1,                            
	    linewidth=0,
	    edgecolor=None,
	    facecolor="#bdbdbd",
	    alpha=0.5
		)
	ax.add_patch(rect)

	# Trapezoid connecting to bottom subplot
	points = [
	    (gene_start, -2),
	    (gene_end, -2),
	    (zoom_end, 0),
	    (zoom_start, 0)
		]
	trapezoid = patches.Polygon(points, closed=True, edgecolor=None, facecolor="#bdbdbd", alpha=0.5)
	ax.add_patch(trapezoid)



# The layout will change when writing to a file
def rainbow_text(x, y, ls, lc, ax, **kw):
	"""Render different colors for texts"""

	import matplotlib.transforms as transforms
	
	t = ax.transData
	renderer = ax.figure.canvas.get_renderer()
	
	total_width = 0
	dummy_texts = []
	for s, c in zip(ls, lc):
		text = ax.text(0, 0, s, color=c, transform=t, **kw)
		text.draw(renderer)
		ex = text.get_window_extent(renderer=renderer)
		total_width += ex.width
		dummy_texts.append(text)
	
	for txt in dummy_texts:
		txt.remove()
	
	t = transforms.offset_copy(t, x=-total_width, units='dots')
	
	for s, c in zip(ls, lc):
		text = ax.text(x, y, s, color=c, transform=t, **kw)
		text.draw(renderer)
		ex = text.get_window_extent(renderer=renderer)
		t = transforms.offset_copy(t, x=ex.width, units='dots')



def expand_genomic_pos(r, size):
	from genomictools import GenomicPos
	
	r = GenomicPos(r)
	s = size - len(r)
	return GenomicPos(r.name, r.start - (s // 2), r.stop + (s // 2 + s % 2))


	
#--------------------------------------------------------------------------------------------------------
# Fi-NeMo
#--------------------------------------------------------------------------------------------------------

def finemo_preprocessing(one_hot, contrib, output_prefix, width="1000"):
	"""
	Prepare input format required for Fi-NeMo
	"""
	# Preprocessing commands do not require GPU.
	# finemo extract-regions-modisco-fmt -s <sequences> -a <attributions> -o <out_path> [-w <region_width>]
	# -s/--sequences: A .npy or .npz file containing one-hot encoded sequences.
	# -a/--attributions: One or more .npy or .npz files of hypothetical contribution scores, with paths delimited by whitespace. Scores are averaged across files.
	# -o/--out-path: The path to the output .npz file.
	# -w/--region-width: The width of the input region centered around each peak summit. Default is 1000.
	
	commands = ["finemo extract-regions-modisco-fmt",
				"-s", one_hot,
				"-a", contrib,
				"-o", output_prefix,
				"-w", width
			   ]
	print(" ".join(commands))



def finemo_call_hits(regions, modisco, outdir, peaks, alpha="0.7"):
	"""
	Identify hits in input regions using TFMoDISCo CWM's.
	"""
	# Usage: finemo call-hits -r <regions> -m <modisco_h5> -o <out_dir> [-p <peaks>] [-t <cwm_trim_threshold>] [-a <alpha>] [-b <batch_size>] [-J]
	# -r/--regions: A .npz file of input sequences and contributions. Created from the above "preprocessing" step.
	# -m/--modisco-h5: A tfmodisco-lite output H5 file of motif patterns.
	# -o/--out-dir: The path to the output directory.
	# -p/--peaks: A peak regions file in ENCODE NarrowPeak format, exactly matching the regions specified in --regions.
	# -t/--cwm-trim-threshold: The threshold to determine motif start and end positions within the full CWMs. Default is 0.3.
	# -a/--alpha: The L1 regularization weight. Default is 0.7.
	# -b/--batch-size: The batch size used for optimization. Default is 2000.
	# -J/--compile: Enable JIT compilation for faster execution. This option may not work on older GPUs.
	
	# The -a/--alpha controls the sensitivity of the hit-calling algorithm, with higher values resulting in fewer but more confident hits. This parameter represents the minimum correlation between a query contribution score window and a CWM to be considered a hit. The default value of 0.7 typically works well for chromatin accessiblity data. ChIP-Seq data may require a lower value (e.g. 0.6).
# The -t/--cwm-trim-threshold parameter sets the maximum relative contribution score in trimmed-out CWM flanks. If you find that motif flanks are being trimmed too aggressively, consider lowering this value. However, a too-high value may result in closely-spaced motif instances being missed.
	# Set -b/--batch-size to the largest value your GPU memory can accommodate. If you encounter GPU out-of-memory errors, try lowering this value.

	import os
	
	if not os.path.exists(outdir):
		os.makedirs(outdir, exist_ok=True)
	commands = ["finemo call-hits",
				"-r", regions,
				"-m", modisco,
				"-o", outdir,
				"-p", peaks,
				"-a", alpha,
				"-J"
			   ]
	print(" ".join(commands))



#--------------------------------------------------------------------------------------------------------
# Motif enrichment (HOMER)
#--------------------------------------------------------------------------------------------------------

def generate_homer_input(es, outputfile):
	"""
	Generate input files conforming to HOMER's format
	"""
	from biodata.delimited import DelimitedWriter
	
	with DelimitedWriter(outputfile) as dw:
		for e in es:
			chrom, start, end = e.split("_")
			dw.write([chrom, start, end, "_".join([chrom, start, end]), ".", "+"])



def run_homer(homer_dir, target, bg, outdir, motif_file=None, denovo=True):
	"""
	Run HOMER for motif enrichment
	"""
	import subprocess
	from os.path import exists
	
	# HOMER Motif Database (http://homer.ucsd.edu/homer/motif/motifDatabase.html): This database is maintained as part of HOMER and is mostly based on the analysis of public ChIP-Seq data sets. These motifs are often referred to in the HOMER software as 'known' motifs since their degeneracy thresholds have been optimized by HOMER, unlike motifs found in JASPAR or other public databases.

	if exists(outdir):
		subprocess.run(f"rm -r {outdir}", shell=True)
		
	commands = [f"{homer_dir}bin/findMotifsGenome.pl",
				 target,
				 "hg38",
				 outdir,
				 "-size given",
				 "-bg", bg
				]
	if not denovo:	
		commands.append("-nomotif")
	if motif_file:
		commands.append(f"-mknown {motif_file}")
		
	subprocess.run(" ".join(commands), shell=True)



#--------------------------------------------------------------------------------------------------------
# Motif scanning (FIMO)
#--------------------------------------------------------------------------------------------------------

def generate_fimo_input(snps, fdict, outputfile):
	"""Generate fasta file for wt and mt sequences (100bp centered on SNP)
	"""
	from biodata.delimited import DelimitedWriter
	
	with DelimitedWriter(outputfile) as dw:
		for snp in snps:
			chrom, snp_pos, rsid, ref, alt = snp
			start, end = int(snp_pos)-50, int(snp_pos)+50
			wt_seq = fdict[chrom][start:end].seq.upper()
			if wt_seq[50] == ref:
				dw.write([f">{rsid}_wt"])
				dw.write([wt_seq])
				dw.write([f">{rsid}_mt"])
				mt_seq = wt_seq[:50] + alt + wt_seq[51:]
				dw.write([mt_seq])



def run_fimo(fimo_dir, motif_file, seq_file, outputfile):
	"""
	Run FIMO for motif scanning
	"""
	import subprocess

	commands = [f"{fimo_dir}src/fimo", 
				 "--skip-matched-sequence",
				 "--verbosity 1",
				 motif_file, 
				 seq_file,
				 ">", outputfile
				 ]
	subprocess.run(" ".join(commands), shell=True)



#--------------------------------------------------------------------------------------------------------
# Pathway enrichment (Enrichr)
#--------------------------------------------------------------------------------------------------------

def run_enrichr(ks, gmt, DE_genes, bg_genes):
	"""Run gene pathway enrichment analysis"""
	import gseapy as gp
	import pandas as pd
	import numpy as np

	dfs = []
	for k in ks:
		enr = gp.enrich(gene_list=list(DE_genes[k]),
						background=list(bg_genes[k]),
						 gene_sets=gmt,
						 verbose=True
						 )
		df = enr.results
		df["Group"] = str(k)
		df["Gene ratio"] = df['Overlap'].apply(lambda x: float(x.split('/')[0]) / float(x.split('/')[1]))
		df["Gene count"] = df['Overlap'].apply(lambda x: float(x.split('/')[1]))
		df = df[["Group", "Term", "Gene ratio", "Gene count", "Adjusted P-value"]].copy()
		df["-log10(padj)"] = -np.log10(df["Adjusted P-value"])
		df = df.sort_values(by="-log10(padj)", ascending=False)
		dfs.append(df)
	df_merge = pd.concat(dfs)
	
	return df_merge



#--------------------------------------------------------------------------------------------------------
# Motif logo
#--------------------------------------------------------------------------------------------------------

def parse_meme(inputfile):
	"""
	Goal:
	Parse meme file to get letter-probability matrix for each motif

    Params:
	inputfile: /home/yc2553/TRE_directionality/databases/motifs/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
	"""

	import pandas as pd
	
	current_motif = None
	motif_data = []
	with open(inputfile) as f:
		for line in f:
			line = line.strip()
			if line.startswith("MOTIF"):
				motif_id = line.split(" ")[1]
				current_motif = {"id": motif_id, "name": motif_id.split("_")[0], "matrix": []}
			elif current_motif is not None:
				if not line.startswith("letter") and not line.startswith("URL"):
					probabilities = [float(prob) for prob in line.split()]
					current_motif["matrix"].append(probabilities)
				elif line.startswith("URL"):
					motif_data.append(current_motif)
					current_motif = None
	dfs = {}
	for n in range(len(motif_data)):
		dfs[motif_data[n]["id"]] = {"name": motif_data[n]["name"], "df": pd.DataFrame(motif_data[n]["matrix"], columns=["A", "C", "G", "T"], index=[n for n in range(1, len(motif_data[n]["matrix"])+1)])}

	return dfs



def ppm2im(ppm):
	"""
	Goal:
	Convert position probability matrix to information matrix (bits).
	"""

	import numpy as np
	import pandas as pd
	
	exp = 4*(-0.25)*np.log2(0.25)
	results = []
	for index, row in ppm.iterrows():
		obs = sum([-row[c]*np.log2(row[c]) if row[c] > 0 else 0 for c in ppm.columns])
		results.append([(exp-obs)*row[c] for c in ppm.columns])
	df = pd.DataFrame(results, index=ppm.index, columns=ppm.columns)
	return df



# Get codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
def plot_a(ax, base, left_edge, height, color):

	import numpy as np
	import matplotlib
	
	a_polygon_coords = [
		np.array([
			[0.0, 0.0],
			[0.5, 1.0],
			[0.5, 0.8],
			[0.2, 0.0],
		]),
		np.array([
			[1.0, 0.0],
			[0.5, 1.0],
			[0.5, 0.8],
			[0.8, 0.0],
		]),
		np.array([
			[0.225, 0.45],
			[0.775, 0.45],
			[0.85, 0.3],
			[0.15, 0.3],
		])
	]

	for polygon_coords in a_polygon_coords:
		ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords + np.array([left_edge, base])[None, :]), facecolor=color, edgecolor=color))



# Get codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
def plot_c(ax, base, left_edge, height, color):

	import numpy as np
	import matplotlib

	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height, facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height, facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height, facecolor='white', edgecolor='white', fill=True))



# Get codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
def plot_g(ax, base, left_edge, height, color):
	
	import numpy as np
	import matplotlib
	
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height, facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height, facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height, facecolor='white', edgecolor='white', fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height, facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height, facecolor=color, edgecolor=color, fill=True))



# Get codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
def plot_t(ax, base, left_edge, height, color):

	import numpy as np
	import matplotlib
	
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base], width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height], width=1.0, height=0.2 * height, facecolor=color, edgecolor=color, fill=True))



# Get codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}
def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):

	import numpy as np
	import matplotlib
	
	if len(array.shape) == 3:
		array = np.squeeze(array)
	assert len(array.shape) == 2, array.shape
	if array.shape[0] == 4 and array.shape[1] != 4:
		array = array.transpose(1, 0)
	assert array.shape[1] == 4
	max_pos_height = 0.0
	min_neg_height = 0.0
	heights_at_positions = []
	depths_at_positions = []
	for i in range(array.shape[0]):
		acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
		positive_height_so_far = 0.0
		negative_height_so_far = 0.0
		for letter in acgt_vals:
			plot_func = plot_funcs[letter[0]]
			color = colors[letter[0]]
			if letter[1] > 0:
				height_so_far = positive_height_so_far
				positive_height_so_far += letter[1]
			else:
				height_so_far = negative_height_so_far
				negative_height_so_far += letter[1]
			plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
		max_pos_height = max(max_pos_height, positive_height_so_far)
		min_neg_height = min(min_neg_height, negative_height_so_far)
		heights_at_positions.append(positive_height_so_far)
		depths_at_positions.append(negative_height_so_far)
	
	for color in highlight:
		for start_pos, end_pos in highlight[color]:
			assert start_pos >= 0.0 and end_pos <= array.shape[0]
			min_depth = np.min(depths_at_positions[start_pos:end_pos])
			max_height = np.max(heights_at_positions[start_pos:end_pos])
			ax.add_patch(
				matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
											 width=end_pos - start_pos,
											 height=max_height - min_depth,
											 edgecolor=color, fill=False))
	
	ax.set_xlim(-length_padding, array.shape[0] + length_padding)
	ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
	height_padding = max(abs(min_neg_height) * (height_padding_factor),
						 abs(max_pos_height) * (height_padding_factor))
	ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
	return ax



# Modify codes from Taskiran_et_al_Supplemental_Code/Fly/utils.py
def plot_weights(array, ax,
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):

	import matplotlib
	
	y = plot_weights_given_ax(ax=ax, array=array,
                              height_padding_factor=height_padding_factor,
                              length_padding=length_padding,
                              subticks_frequency=subticks_frequency,
                              colors=colors,
                              plot_funcs=plot_funcs,
                              highlight=highlight)
	return ax



