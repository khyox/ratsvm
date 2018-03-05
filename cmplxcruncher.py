#!/usr/bin/env python3
#
# cmplxcruncher - DLS team - JMMM
#
# Complex-systems analyzer Python3 code. PEP8 compliant.
#
# Require packages: matplotlib, numpy, pandas >= 0.15.0 with openpyxl,
#                   scipy, xlrd > 0.9

# Packages
import argparse
import cmplxcruncher as cc
from cmplxcruncher.plots import enableDevMode, ChkDir
from cmplxcruncher.sessions import set_release


# INTERNAL CONSTANTS
# Defaults directories
_DEFAULT_DATA_DIR = './data'
_DEFAULT_RESULTS_DIR = 'results'
_DEFAULT_FREQLIM_SUFFIXDIR = ''


# > MAIN

if __name__ == '__main__':
    # Argument Parser Configuration
    parser = argparse.ArgumentParser(
        description='Complex systems analyzer Python3 code',
        epilog='cmplxcruncher - DLS team - by J.Mn.Marti - ' + cc._verdata
    )
    parser.add_argument(
        '-V', '--version',
        action='version',
        version='cmplxcruncher.py ver. ' + cc.__version__ + ' - ' + cc._verdata
    )
    parser.add_argument(
        '-p', '--path',
        action='store',
        default=_DEFAULT_DATA_DIR,
        help=('path of the data files (if not present, \'' +
              _DEFAULT_DATA_DIR + '\' will be tried)')
    )
    parser.add_argument(
        '-x', '--fmax',
        type=float,
        metavar='F',
        action='store',
        default=1,
        help=('maximum relative frequency allowed ' +
              '(only for the sum over time of an item), from 0 to 1 (default)')
    )
    parser.add_argument(
        '-z', '--fmin',
        type=float,
        metavar='F',
        action='store',
        default=0,
        help=('minimum relative frequency allowed (for any single item@time' +
              ' and for the sum over time), from 0 (default) to 1')
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help=('force go ahead with the analysis even if repeated elements' +
              ' are found in the data imported')
    )
    parser.add_argument(
        '-g', '--debug',
        action='store_true',
        help=('produce debugging info  (useful mainly for sequential mode)')
    )
    parser.add_argument(
        '-c', '--core',
        action='store_true',
        help=('do further analysis related with the core of elements')
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help=('produce verbose output  (combine with --debug for further info)')
    )
    # Parser group: Mode
    groupMode = parser.add_mutually_exclusive_group(required=True)
    groupMode.add_argument(
        '-i', '--interactive',
        action='store_true',
        help=('run the code in interactive mode')
    )
    groupMode.add_argument(
        '-a', '--automatic',
        action='store',
        choices={'png', 'pdf', 'svg', 'ps', 'eps'},
        help=('run the code in automatic mode generating plots with the' +
              ' given format')
    )
    groupMode.add_argument(
        '-t', '--test',
        action='store',
        choices={'png', 'pdf', 'svg', 'ps', 'eps'},
        help=('run the code in test mode generating plots with given format')
    )
    # Parser group: Plot
    groupPlot = parser.add_mutually_exclusive_group(required=False)
    groupPlot.add_argument(
        '-l', '--lessplot',
        action='store_true',
        help='if present draw only std VS mean plots'
    )
    groupPlot.add_argument(
        '-m', '--moreplot',
        action='store_true',
        help=('if present, draw all the available plots')
    )
    # Parser group: Results
    groupResults = parser.add_mutually_exclusive_group(required=False)
    groupResults.add_argument(
        '-n', '--nodirs',
        action='store_true',
        help=('no directories will be created to store the results' +
              ' of the analysis (legacy behaviour of the beta releases)')
    )
    groupResults.add_argument(
        '-r', '--results',
        action='store',
        default=_DEFAULT_RESULTS_DIR,
        help=('path of the results directory relative to the data directory' +
              ' (if not present, "' + _DEFAULT_RESULTS_DIR +
              '" will be tried)')
    )
    # Parse arguments
    flgs = parser.parse_args()
    datapath = flgs.path

    # Program Header
    print('\n=== cmplxcruncher === v' + cc.__version__ + ' === ' +
          cc._verdata + ' === by DLS team ===')
    if(cc._devflag):
        print('\n>>> WARNING! THIS IS JUST A DEVELOPMENT SUBRELEASE.' +
              ' USE AT YOUR OWN RISK!')
    if(not cc._useLaTeX):
        print('\n>>> WARNING! PyLaTeX and/or uncertainties python libraries ' +
              'not found.\n\t\t LaTeX output will not be generated!')
    # Check the value for the fmin argument
    if (flgs.fmin < 0 or flgs.fmin > 1):
        raise Exception('ERROR! Invalid value for the fmin argument' +
                        ' (expected float between 0 and 1)')
    # Check the value for the fmax argument
    if (flgs.fmax < 0 or flgs.fmax > 1):
        raise Exception('ERROR! Invalid value for the fmax argument' +
                        ' (expected float between 0 and 1)')
    # Check fmin vs fmax
    if (flgs.fmin >= flgs.fmax):
        raise Exception('ERROR! fmin cannot be equal or higher than fmax')

    # Select the results directory and check for existence and permissions
    if flgs.nodirs:
        respath = ''
    else:
        # If the results dir is the default but a freq limit is required,
        # then specify special results dir
        if ((flgs.results == _DEFAULT_RESULTS_DIR) and
                (flgs.fmin > cc._EPS or flgs.fmax < (1 - cc._EPS))):
            respath = flgs.results + _DEFAULT_FREQLIM_SUFFIXDIR + '/'
        else:
            respath = flgs.results + '/'
        if (ChkDir(datapath + '/' + respath)):
            respath = ''
            print('          Warning: the data root directory will be tried' +
                  ' instead, for writing results...')

    # Select the dev mode in plot module
    if cc._devflag or flgs.debug:
        enableDevMode()

    # Set release in sessions module
    set_release(cc.__version__, cc._verdata)

    # Select the running mode (test, interactive or automatic)
    if flgs.test:
        enableDevMode()
        from cmplxcruncher.sessions import TestSession
        TestSession(flgs, respath)
    elif flgs.interactive:
        from cmplxcruncher.sessions import InteractiveSession
        InteractiveSession(flgs, respath)
    else:
        from cmplxcruncher.sessions import AutomaticSession
        AutomaticSession(flgs, respath)

# < MAIN
