#!/usr/bin/env python3
#
# Python 3.X site-specific launcher for LMAT run_rl.sh (1st step)
#

import argparse
import os
import sys
import errno
import subprocess as sp

# pyLMAT release information
__version__ = '0.0.13_ratsvm'
_verdata = 'Mar 2018'
_devflag = True

# Predefined internal constants
_PATH = './unmap/'
_BDIR = '~/ratsvm/'
_FULL = '~/ratsvm/LMAT/'
_LMAT = '~/ratsvm/LMAT/'
_RTIN = 'runtime_inputs/'

# Argument Parser Configuration
parser = argparse.ArgumentParser(
    description='pythonic LMAT launcher',
    epilog='pyLMAT_rl for RATSVM - JMMM - ' + _verdata
)
parser.add_argument(
    '-v', '--version',
    action='version',
    version='pyLMAT_rl.py release ' + __version__ + ' - ' + _verdata
)
parser.add_argument(
    '-p', '--path',
    action='store',
    default=_PATH,
    help=('relative path of the data files (if omitted, \'' +
          _PATH + '\' will be tried)')
)
parser.add_argument(
    '-b', '--bdir',
    action='store',
    metavar='PATH',
    default=_BDIR,
    help=('base directory (if omitted, \'' +
          _BDIR + '\' will be tried)')
)
parser.add_argument(
    '-f', '--fulldbdir',
    action='store',
    metavar='PATH',
    default=_FULL,
    help=('path of the full LMAT database lmat-4-14.20mer.db (if omitted, \'' +
          _FULL + '\' will be tried)')
)
parser.add_argument(
    '-l', '--lmat',
    action='store',
    metavar='PATH',
    default=_LMAT,
    help=('path of the LMAT installation (if omitted, \'' +
          _LMAT + '\' will be tried)')
)
parser.add_argument(
    '-t', '--threads',
    action='store',
    default='64',
    help=('number of OpenMP threads to use (64 by default)')
)
parser.add_argument(
    '-m', '--minscore',
    action='store',
    default='0',
    help=('minimum score assigned to read for it to be included in binning' +
          ' (0 by default)')
)
# Parser group: sequencing
groupSeq = parser.add_mutually_exclusive_group(required=True)
groupSeq.add_argument(
    '-w', '--wgs',
    action='store_true',
    default=True,
    help=('suppose Whole Genome Shotgun sequencing')
)
groupSeq.add_argument(
    '-s', '--s16',
    action='store_true',
    default=False,
    help=('suppose 16S sequencing instead of WGS')
)

# Parse arguments
args = parser.parse_args()
path = args.path
bdir = args.bdir
s16 = '_16S' if args.s16 else ''

# Program Header
print('\n=-= pyLMAT_rl =-= v' + __version__ + ' =-= ' +
      _verdata + ' =-= JMMM =-=')
if(_devflag):
    print('\n>>> WARNING! THIS IS JUST A DEVELOPMENT SUBRELEASE.' +
          ' USE AT YOUR OWN RISK!')

# LMAT script and options
_pgrm = 'run_rl.sh'
_db = '--db_file=' + args.fulldbdir + '/lmat-4-14.20mer.db'
_fst = '--query_file='
_odir = '--odir='
_nthr = '--threads=' + args.threads
_ovw = '--overwrite'
_mins = '--min_score=' + args.minscore
_null = ('--nullm=' + os.path.join(args.lmat, _RTIN, 
         'lmat-4-14.20mer.16bit.g200.grand.with-adaptors.db.null_lst.txt'))

# Preparing environment for LMAT
lmat_env = os.environ.copy()
lmat_env["PATH"] = os.path.join(args.lmat, 'bin') + ":" + lmat_env["PATH"]
lmat_env["LIBRARY_PATH"] = os.path.join(args.lmat, 'lib') + ":" + lmat_env["LIBRARY_PATH"]
lmat_env["LD_LIBRARY_PATH"] = os.path.join(args.lmat, 'lib') + ":" + lmat_env["LD_LIBRARY_PATH"]
lmat_env["CPLUS_INCLUDE_PATH"] = os.path.join(args.lmat, 'include') + ":" + lmat_env["CPLUS_INCLUDE_PATH"]
lmat_env["LMAT_DIR"] = os.path.join(args.lmat, _RTIN)  # this is critical

# Example: LMAT/bin/run_rl.sh --db_file=/fastdisk/lmat-4-14.20mer.db --min_score=0 --threads=32 --nullm=/home/martijm/ratsvm/LMAT/runtime_inputs/lmat-4-14.20mer.16bit.g200.grand.with-adaptors.db.null_lst.txt --query_file=./unmap/day000_2_unmap.fastq --odir=/home/martijm/ratsvm/unmap/test

# Main loop
for root, dirs, files in os.walk(path):
    for file in files:
        argums = [_pgrm, _db, _mins, _nthr, _null]
        fileNOext = os.path.splitext(file)[0]
        outdir = os.path.join(bdir, path, fileNOext + s16)
        print('\nStoring results in the dir: ', outdir)
        try:
            os.makedirs(outdir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        argums.append(_fst + os.path.join(path, file))
        argums.append(_odir + outdir)
        print('\tLaunching subprocess: ', argums)
        sys.stdout.flush()
        try:
            sub = sp.Popen(argums, env=lmat_env)
            sub.wait()
        except (PermissionError):
            print('\nERROR! Unable to launch ' + repr(_pgrm))
            print('TIP: Check if the path to LMAT bin is right\n')
            raise
