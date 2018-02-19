#!/usr/bin/env python3
"""
Get all the paired-ends FASTQ files associated with run accession (SRR) list
"""

import argparse
import itertools
import os.path
import subprocess as sp
import sys

STK_PATH = '/Users/martijm/local/sratoolkit'
STK_DUMPER = 'fastq-dump'
FLAG_SPLITFILES = '--split-files'
FLAG_MRLEN = '--minReadLen'
FLAG_FASTA = '--fasta'
FLAG_VERBO = '-v'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-s', '--sratoolkit',
        action='store',
        metavar='PATH',
        default=STK_PATH,
        help='Path for NCBI SRA Toolkit'
    )
    parser.add_argument(
        '-f', '--file',
        action='store',
        metavar='FILE',
        help='File with a list of run accession (SRR) codes. '
             'This option has preference over the standard input.'
    )

    args = parser.parse_args()
    program = os.path.join(args.sratoolkit, 'bin', STK_DUMPER)

    # Release information
    __version__ = '0.1.1'
    __date__ = 'Feb 2018'

    # Program header
    print(f'\n=-= {sys.argv[0]} =-= v{ __version__} =-= {__date__} =-=\n')

    if args.file:
        srr_file = open(args.file, 'r')
    else:
        srr_file = sys.stdin

    subs = {}
    print(f'Launching retrieval {program} processes:')
    for line in srr_file:
        srr = line.rstrip()
        argums = [program, FLAG_SPLITFILES, srr]
        sub = sp.Popen(argums, stdout=sp.PIPE, stderr=sp.PIPE,
                       universal_newlines=True)
        print(f'\tRetrieving {srr} with subprocess with PID {sub.pid}')
        subs[srr] = sub

    print(f'\nWaiting for {len(subs)} {STK_DUMPER} processes to complete...')
    progress = itertools.cycle(r'-\|/')
    finished = {srr: False for srr in subs.keys()}
    while not all(finished.values()):
        running_subs = {k: v for k, v in filter(lambda i: not finished[i[0]],
                                                subs.items())}
        for srr, sub in running_subs.items():
            outs = None
            errs = None
            print('\r\033[95m{}\033[0m [{:.2%}] full retrieved  '.format(
                next(progress), 1 - len(running_subs) / len(subs)), end='')
            try:
                outs, errs = sub.communicate(timeout=1)
            except sp.TimeoutExpired:
                pass
            except KeyboardInterrupt:
                print('Killed by user! ', end='')
                sub.kill()
                outs, errs = sub.communicate()
                finished[srr] = True
            if outs or errs:
                print(f'Output from retrieval of {srr} (PID {sub.pid}):')
                print(outs, errs)
            if sub.returncode:
                print(f'Retrieving {srr} finished with '
                      f'exit code {sub.returncode}')
                finished[srr] = True
    print(f'All {STK_DUMPER} processes finished!')


if __name__ == '__main__':
    main()
