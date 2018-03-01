#!/usr/bin/env fish
# JMMM - rel 0.2

function help_exit
    echo "Usage:  [options] bt2database"
    echo "Options:"
    echo "-s/--samtls_path PATH : samtools path"
    echo "-p/--proc PROC : Number of processors"
    echo "-h/--help: Display this help"
    echo "Argument: for example, hg19plusRNA"
    exit 1
end

set -l options (fish_opt -s h -l help)
set options $options (fish_opt -s p -l proc --required-val)
set options $options (fish_opt -s s -l samtls_path --required-val)
argparse --name launch_samtools --min-args 1 --max-args 1 $options -- $argv
if test ! $status -eq 0; or set -q _flag_h
    help_exit
end

# Set default values
set -q _flag_p; or set -l _flag_p 32
set -q _flag_s; or set -l _flag_s ~/ratsmv/SAMtools

# Prepend to path 
set -x PATH $_flag_s/bin $PATH 

# Echo vars
echo "Options:" $_flag_s/bin $_flag_p 
echo "Database:" $argv

#module purge
#module load gcc/gcc-6.2.0_SOM4 bio/samtools
set fnames day000_1 day182_0 day184_1 day852_1 day854_1 day880_1 day882_1 day000_2 day182_1 day851_1 day852_2 day855_1 day880_2 day882_B day180_1 day182_2 day851_2 day853_1 day879_1 day881_1 day883_1 day181_1 day183_1 day852_0 day853_2 day879_2 day881_2
for fname in $fnames
	echo ">>> Next, get unmapped reads for $fname..."
	echo "samtools view -f 13 $fname _$argv.sam | samtools sort - -n -O sam -@ $_flag_p | samtools fastq - -1 $fname _unmap_R1.fastq -2 $fname _unmap_R2.fastq"
	samtools view -f 13 "$fname"_"$argv".sam | samtools sort - -n -O sam -@ "$_flag_p" | samtools fastq - -1 "$fname"_unmap_R1.fastq -2 "$fname"_unmap_R2.fastq
	echo "<<< "
end
