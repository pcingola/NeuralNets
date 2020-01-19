
#-------------------------------------------------------------------------------
#
# This R script downloads TCGA datasets
#
# Requirements:
#
#	- R version 3.6 or higher
#		- Run command line: `R --version`
#
#	- R package BioCondctor (https://bioconductor.org/install/)
#		You can install BioConductor by running
#		```
#		if (!requireNamespace("BiocManager", quietly = TRUE))
#			install.packages("BiocManager")
#		BiocManager::install()
#		```
#
#	- BioConductor package RTGCA (https://bioconductor.org/packages/release/bioc/html/RTCGA.html)
#		You can install RTCGA by running
#		```
#		if (!requireNamespace("BiocManager", quietly = TRUE))
#			install.packages("BiocManager")
#		BiocManager::install("RTCGA")
#		```
#
#	- R package tidyverse: ``
#
#																Pablo Cingolani
#-------------------------------------------------------------------------------

library('RTCGA')
library('tidyverse')

#-------------------------------------------------------------------------------
# Download any matching dataset
#-------------------------------------------------------------------------------
downloadMatchingTcga <- function(cohort, regex, datasets) {
	dataset <- grep(regex, datasets, value=TRUE, ignore.case=TRUE)
	if( !is_empty(dataset) ) {
		cat("Matching datasets', length(dataset), ': [", paste(dataset, ','), ']\n')
		for(ds in dataset) {
			cat("\tDownloading dataset:", ds, '\n')
			downloadTCGA(cohort, dataSet=ds, destDir='.')
		}
	}
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

# Get cohort names
infotcga <- infoTCGA()
cohorts <- infotcga$Cohort

# Download data for every cohort
for(cohort in cohorts) {
	cat("Cohort:", cohort, '\n')
	datasets <- checkTCGA('DataSets', cohort)$Name
	downloadMatchingTcga(cohort, 'Merge_Clinical', datasets)	# Clinical data
	downloadMatchingTcga(cohort, 'Mutation_Packager_Calls', datasets)	# Mutations
	downloadMatchingTcga(cohort, 'RSEM_genes_normalized', datasets)	# RNA-Seq data
	downloadMatchingTcga(cohort, 'segmented_scna_minus_germline_cnv_hg19', datasets)	# CNA, copy number alterations (a.k.a. CNVs)
}
