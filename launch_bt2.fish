#!/usr/bin/env fish 
# JMMM - rel. 0.3

function help_exit
    echo "Usage:  [options] bt2database"
    echo "Options:"
    echo "-b/--bt2_path PATH : Bowtie2 path"
    echo "-d/--db_path PATH : Database path"
    echo "-p/--proc PROC : Number of processors"
    echo "-v/--very-sensitive : Enable very sensitive mode"
    echo "Argument: for example, hg19plusRNA"
    exit 1
end

set -l options (fish_opt -s h -l help)
set options $options (fish_opt -s v -l very-sensitive)
set options $options (fish_opt -s p -l proc --required-val)
set options $options (fish_opt -s d -l db_path --required-val)
set options $options (fish_opt -s b -l bt2_path --required-val)
argparse --name launch_bt2 --min-args 1 --max-args 1 $options -- $argv 
if test ! $status -eq 0; or set -q _flag_h
    help_exit
end

# Set default values
set -q _flag_p; or set -l _flag_p 64
set -q _flag_d; or set -l _flag_d ~/ratsvm/HSS
set -q _flag_b; or set -l _flag_b ~/ratsvm/bowtie2

# Set aux variables values
set -q _flag_v; and set -g opts --very-sensitive

# Echo vars
echo "Options:" $opts $_flag_p $_flag_d $_flag_b
echo "Database:" $argv

set fnames day000_1  day182_0  day184_1  day852_1  day854_1  day880_1  day882_1 day000_2  day182_1  day851_1  day852_2  day855_1  day880_2  day882_B day180_1  day182_2  day851_2  day853_1  day879_1  day881_1  day883_1 day181_1  day183_1  day852_0  day853_2  day879_2  day881_2
for fname in $fnames
	echo ">>> Next, against DB=$argv will align reads in $fname..."
	echo "$_flag_b"/bowtie2 -t -x "$_flag_d/$argv" -1 "$fname"_R1.fastq -2 "$fname"_R2.fastq -S "$fname"_"$argv".sam -p "$_flag_p" "$opts"
	eval "$_flag_b"/bowtie2 -t -x "$_flag_d/$argv" -1 "$fname"_R1.fastq -2 "$fname"_R2.fastq -S "$fname"_"$argv".sam -p "$_flag_p" "$opts"
	echo "<<< "
end
