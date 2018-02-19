# Robust Analysis of Time Series in Virome Metagenomics

### Overview

This repository contains scripts used in the chapter _Robust Analysis of Time Series in Virome Metagenomics_, of the book _The Human Virome: Methods and Protocols_, _MiMB Series_ of Springer.

### runs2fastq.py

This python script gets all the paired-ends FASTQ files associated with a list with run accession (SRR) codes. 

The list of SRR codes (one per line) is read from a file or from the standard input.

Example of `runs2fastq.py` as part of a pipe using the script `sra2runacc.py` from the package [https://github.com/jordibc/entrez](entrez) to get the split FASTQ files for all the runs of a study in SRA (_SRP021107_ in the example):

```
sra2runacc.py -s SRP021107 | runs2fastq.py
```


