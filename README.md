# Robust Analysis of Time Series in Virome Metagenomics

## Overview

This repository contains scripts used in the chapter _Robust Analysis of Time Series in Virome Metagenomics_, of the book _The Human Virome: Methods and Protocols_, _Methods in Molecular Biology_ series of [Springer](http://www.springer.com/).

## runs2fastq.py

This python script gets all the paired-ends FASTQ files associated with a list with run accession (SRR) codes. 

The list of SRR codes (one per line) is read from a file or from the standard input.

Example of `runs2fastq.py` as part of a pipe using the script `sra2runacc.py` from the package [entrez](https://github.com/jordibc/entrez) to get the split FASTQ files for all the runs of a study in SRA (_SRP021107_ in the example):

```
sra2runacc.py -s SRP021107 | runs2fastq.py
```

## make_hg19plusRNA.sh

The `make_hg19plusRNA.sh` script downloads the UCSC hg19 human reference. Those sequences join the mRNA and ncRNA sequences if they are already downloaded in the same location. Then, `bowtie2-build` is called in order to build the bowtie2 reference database. Please see the protocol for detailed usage information.

## launch_bt2.fish

This script manages the alignment of the reads against the extended hg19 human genome reference prepared with `make_hg19plusRNA.sh` or any other processed database. `launch_bt2.fish` requires just one positional argument: the name of the bowtie2 alignment database. It also accepts various options. Run it with the  -h/--help flag or consult the protocol notes for more info.

## launch_samtools.fish

The `launch_samtools.fish` script can be used to obtain FASTQ files for sequences that did not align against the extended hg19 human genome after running `launch_bt2.fish`. It also requires the name of the bowtie2 alignment database as positional argument. This script accepts various options to fine tune the process, so run it with the  -h/--help flag or check the protocol notes for detailed information.

## pyLMAT_rl.py and pyLMAT_cs.py 

These scripts are convenient wrappers for the taxonomic classification module (`pyLMAT_rl.py`) and content summarization (`pyLMAT_cs.py`) step of [LMAT (Livermore Metagenomic Analysis Toolkit)](https://github.com/LivGen/LMAT).

## rawlmat2lmat.py and lmat2cmplx.py

The scripts `rawlmat2lmat.py` and `lmat2cmplx.py` perform a two-step translation of the LMAT results for both the taxonomic classification module and content summarization, to the input format of cmplxcruncher.

## cmplxcruncher (beta release)

cmplxcruncher computing kernel is a python tool to analyze the dynamics of ranking processes in metagenomics that is still under development. Currently, it already performs many interactive and automatic analysis with both numeric and graphic results.

____
