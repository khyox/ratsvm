"""
This module provides the processing classes of the package.
"""

# Packages
import sys
import os
import platform
import time
import xlrd
import csv
import itertools as it
from collections import Counter
import multiprocessing as mp
import numpy as np
import pandas as pd
from cmplxcruncher.config import *
from cmplxcruncher.plots import (DoTestPlots, DoAllPlots,
                                 SummaryPlot, ColorSummaryPlot)
# optional packages pylatex and uncertainties
_useLaTeX = True
try:
    import pylatex as ltx
    from uncertainties import ufloat
except ImportError:
    _useLaTeX = False

# Constants
_EPS = np.finfo(np.double).eps
_DF_COLS = [
    'Elements', 'Times', 'AbsFreq_mean', 'AbsFreq_std', 'AbsFreq_sum',
    'V', 'V_err', 'beta', 'beta_err', 'R^2', 'pcorr^2', 'model',
    'xW_V', 'xW_V_err', 'xW_beta', 'xW_beta_err', 'xW_R^2', 'xW_pcorr^2',
    'xW_model',
    'xW_V_stan', 'xW_V_err_stan', 'xW_beta_stan', 'xW_beta_err_stan'
]

# Define "module static" variables __version and __verdata
_version = None
_verdata = None


def set_release(version, verdata):
    """Define 'module static' variables for release"""
    global _version
    _version = version
    global _verdata
    _verdata = verdata


def call_it(instance, fname, *args, **kwds):
    """Indirect caller for instance methods and multiprocessing."""
    # print('call_it to instance='+str(instance)+', fname='+fname)
    return getattr(instance, fname)(*args, **kwds)


class SessionError(Exception):

    """Base class for exceptions in the :mod:`cmplxcruncher.sessions` module"""

    def __init__(self, message):
        super().__init__(message)


class StandardizationNumElmError(SessionError):

    """Raised when trying to standardize with not enough h_ elements."""

    def __init__(self, instance, frame):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Aborted! Not enough h_ elements! '
        )


class Session(object):

    """Base class for the different session classes."""

    def __init__(self):
        pass

    def File2DataFrame(self, filename, sheetname):
        """Get a pandas DataFrame from a Excel/sheet or from a text file."""
        outlog = ''
        if (not filename.endswith('.txt')):
            pd_xlsx = pd.ExcelFile(filename)
            A = pd_xlsx.parse(
                sheetname=sheetname, header=0,
                skiprows=None, index_col=0
            )
        else:
            A = pd.read_table(filename, index_col=0)
        # Traspose the dataframe if needed so time grows with columns
        # (test the label length, and the absolute difference with 1 of
        # the mean of the accumulated values per dimension)
        if (np.mean([len(str(elm)) for elm in A.columns]) >
                np.mean([len(str(elm)) for elm in A.index])):
            if (abs(A.sum(axis='index').mean() - 1) >
                    abs(A.sum(axis='columns').mean() - 1)):
                A = A.T
                outlog += '\t\t WARNING! Auto-transpose of data performed!\n'
        # Drop elements (rows) with any NaN, but warn about!
        _Anull = pd.isnull(A).any(axis='columns')
        _Anullindex = _Anull[_Anull].index  # [_Anull] equival [_Anull == True]
        if (len(_Anullindex)):
            for row in _Anullindex:
                outlog += ('\t \033[91mDANGER!\033[0m To be erased NULL row' +
                           ' labeled "' + str(row) + '"\n')
            A.dropna(axis='index', how='any', inplace=True)
        return(A, outlog)

    def renormalize_DF(self, A, log):
        """Check for suspicious timming data and renormalize in DataFrame."""
        _Asum = A.sum(axis='index')
        # Warn columns which sum is far from 1 (tolerance 1e-2) or negative
        _Awarn = _Asum[(np.abs(_Asum - 1.0) > 1e-2) | (_Asum < _EPS)]
        if len(list(_Awarn)):
            bflagelim = False
            log = (log + '\t\t NOTE: Renormalizing to 1 time registers' +
                   ' (columns) out of prob tol...\n')
            for i, column in enumerate(_Awarn.index):
                if (_Asum[column] < _EPS):
                    A = A.drop(column, axis='columns')
                    log = log + ' [' + str(column) + ' eliminated] '
                    bflagelim = True
                else:
                    A.loc[:, column] = A.loc[:, column].div(_Asum[column])
                    if self.verbose:
                        log = log + ' ' + str(column) + ' '
                if ((i + 0) % 5 == 0) and self.verbose:
                    log = log + '\n\t\t\t'
            if bflagelim or self.verbose:
                log = log + '\n'
        return(A, log, _Asum)

    def split_core_tail_DF(self, C, A, S, inlog):
        """Split the core, the tailless and the tail part of data set."""
        ncols = (A.columns).size
        CORE = 0.2
        # Before: TAIL = 0.6 if ncols < 4 else (0.75 if ncols == 4 else 0.8)
        TAIL = 0.4 if ncols < 4 else (0.5 if ncols == 4 else 0.6)
        data_metalst = []  # List of list of pandas DF/Series & subset name
        index4core = S[CORE <= S['6-ZRF']].index  # Elements not in core
        index4tailless = S[TAIL <= S['6-ZRF']].index  # Elements in tail
        index4onlytail = S[S['6-ZRF'] < TAIL].index  # Elements not it tail
        index_name_lst = [
            [index4core, '_CORE'],
            [index4tailless, '_TAILLESS'],
            [index4onlytail, '_TAIL']
        ]
        outlog = inlog
        for index, subset_name in index_name_lst:
            C_sub, A_sub, S_sub = (X.drop(index, axis='index')
                                   for X in [C, A, S])
            A_sub, ren_outlog, _ = self.renormalize_DF(A_sub, '')
            outlog += ('\t\t\t [' + subset_name + '] dataset:\n' + ren_outlog)
            sstd = A_sub.std(axis='columns')
            s_sub = {
                '1-mean': A_sub.mean(axis='columns'),
                '2-std': sstd,
                '3-skew': A_sub.skew(axis='columns'),
                '4-kurt': A_sub.kurt(axis='columns'),
                '5-sem': sstd / np.sqrt(ncols),
                '6-ZRF': S_sub['6-ZRF']
            }
            S_sub = pd.DataFrame(s_sub, index=A_sub.index)
            data_metalst.append([C_sub, A_sub, S_sub, subset_name])
        data_metalst.append([C, A, S, ''])
        return data_metalst, outlog


class TestSession(Session):

    """Test session."""

    def __init__(self, flgs, respath):
        format = flgs.test
        print('\nChdir to \'' + str(flgs.path) +
              '\' and launching the embedded tests...')
        os.chdir(flgs.path)
        DoTestPlots(respath, format)


class InteractiveSession(Session):

    """"Interactive session."""

    def __init__(self, flgs, respath):
        datapath = flgs.path
        # Obtain the initial data
        filename, sheetname = self.__InteractiveFile(datapath)
        name = os.path.splitext(filename)[0] + '_' + sheetname
        format = self.__InteractiveFormat(filename, sheetname)
        A, outlog = self.File2DataFrame(filename, sheetname)
        print(outlog, end='')

        # Loop for interactive data analysis and plotting
        repeat = True
        while(repeat):
            A, S, Asum, C = self.__InteractiveDataFrame(
                A, flgs.fmin, flgs.fmax, flgs.force)
            data_metalst = [[C, A, S, '']]  # List of list of pandas DF/Series
            if flgs.core:
                data_metalst, outlog = self.split_core_tail_DF(C, A, S, '')
                print(outlog, end='')
            for C_elm, A_elm, S_elm, subset_name in data_metalst:
                localname = name + subset_name
                if subset_name == '':
                    subset_name = 'ALL DATA'
                print('\n Fitting data and plotting figures for ' +
                      subset_name + '...')
                DF = DoAllPlots(
                    A_elm, S_elm, C_elm.sum(axis='index'), C_elm,
                    respath, localname, format, flgs.lessplot,
                    flgs.moreplot, _DF_COLS
                )[0]  # where Asum is calculated in place from C_elm
                print(' The parameters of the fit for ' + subset_name +
                      ' are:')
                try:
                    print(DF[['V', 'V_err', 'beta', 'beta_err', 'R^2']])
                except:
                    print('ERROR! No fit done at all!')
            if format == '':
                print('\nWould you like to repeat something? (ENTER exits)')
                if(not self.__UserChooseBool()):
                    repeat = False
            else:
                repeat = False

    def __UserChooseInt(self, max):
        """
        Captures the user selection (integer range).

        Note: ENTER defaults to 0 (the first choice).

        """
        answer = input('Please, enter a number. Valid range is [0,' +
                       str(max) + '] : ')
        while True:
            try:
                while ((answer != '') and
                       int(answer) not in range(0, max + 1)):
                    answer = input('Invalid option! Valid range is [0,' +
                                   str(max) + '] : ')
                break
            except ValueError:
                answer = input(
                    'You must specify an integer! Valid range is [0,' +
                    str(max) + '] : '
                )
        if (answer == ''):
            return(0)
        else:
            return(int(answer))

    def __UserChooseBool(self):
        """
        Captures the user selection (boolean)

        Note: ENTER defaults to False

        """
        answer = input('Please, enter (y/N): ').lower()
        while True:
            try:
                while ((answer != 'y') and
                       (answer != 'n') and
                       (answer != '')):
                    answer = input('Invalid option! Please, enter (y/N): ')
                break
            except:
                answer = input('Bad choice! Please, enter (y/N): ')
        if (answer == 'y'):
            return(True)
        else:
            return(False)

    def __InteractiveFile(self, datapath):
        """Input the Excel or TXT working file with the right sheet."""
        # Select the file
        dirnames = [datapath, './examples', './data/paper', './']
        filenames = list()
        cwd = os.getcwd()
        for dirname in dirnames:
            try:
                os.chdir(dirname)
                for file in os.listdir('.'):
                    if ((file.endswith('.xls') or
                            file.endswith('.xlsx') or
                            file.endswith('.txt')) and
                            not file.startswith('~') and
                            not file.startswith('cmplxcruncher')):
                        filenames.append(file)
            except (OSError):
                print('WARNING! Cannot change to dir "' + dirname + '"')
            else:
                if filenames:
                    break
                else:
                    print('WARNING! Tried dir "' + dirname +
                          '" but no supported files found...')
            os.chdir(cwd)
        if not filenames:
            raise Exception('ERROR! No Excel or TXT files found.')
        else:
            print('\nWorking directory is "' + dirname + '"')
        filenames = dict(zip(range(len(filenames)), filenames))
        print('\nSelect a file to work with: ')
        print(filenames)
        maxrange = len(filenames) - 1
        filename = filenames[self.__UserChooseInt(maxrange)]
        print('Working file will be: ' + filename)
        sheetname = ''
        if (not filename.endswith('.txt')):
            # Select the sheet
            xlsx = xlrd.open_workbook(filename, on_demand=True)
            sheetnames = xlsx.sheet_names()
            sheetnames = [
                sheetname for sheetname in sheetnames if (sheetname[0] != '_')
            ]
            if (sheetnames == []):
                raise Exception('ERROR! No sheets in the Excel file.')

            sheetnames = dict(zip(range(len(sheetnames)), sheetnames))
            print('\nSelect a sheet to work with:')
            print(sheetnames)
            maxrange = len(sheetnames) - 1
            sheetname = sheetnames[self.__UserChooseInt(maxrange)]
            print('Working sheet will be: ' + sheetname)
        return(filename, sheetname)

    def __InteractiveFormat(self, filename, sheetname):
        """Interactively choose the format of the images."""
        # Select to plot the graphs or to save them
        format = ''
        print('\nWould you like to save the figures to a file?')
        if(self.__UserChooseBool()):
            print('Select the image format:')
            formatnames = {0: 'png', 1: 'pdf', 2: 'ps', 3: 'eps'}
            print(formatnames)
            format = formatnames[self.__UserChooseInt(3)]
        return(format)

    def __InteractiveDataFrame(self, A, fmin, fmax, force):
        """Interactively change a pandas DataFrame."""
        # The user checks the imported data shape
        while(True):
            ncols = (A.columns).size
            nrows = (A.index).size
            print('\nData imported has ' + str(nrows) +
                  ' elements (rows) and ' + str(ncols) +
                  ' time registers (columns).')
            print('Would you like to transpose the data?')
            if(self.__UserChooseBool()):
                A = A.T
            else:
                break

        # Check for repeated elements (rows)
        if (len(A.index.values) != len(set(A.index.values))):
            print('\nCAUTION! The next elements are repeated' +
                  ' the shown number of times:')
            print(
                [(x, y) for x, y in Counter(A.index.values).items() if y > 1])
            if(not force):
                print('Would you like to continue?')
                if(not self.__UserChooseBool()):
                    quit(1)
            print('\n WARNING! You decided to go ahead,' +
                  ' but be aware of inconsistencies! \n')

        # Check for elements with normalized accumulated absolute frequencies
        #  beyond desired limits
        Asum1 = A.sum(axis='columns')
        Asum1 = Asum1.div(Asum1.sum())  # Normalize sum
        # Next IF is the legacy case: check only elements with null frequencies
        if (fmin <= _EPS and fmax >= (1 - _EPS)):
            Awarn = Asum1[Asum1 < _EPS]
        elif (fmin > _EPS and fmax >= (1 - _EPS)):
            Awarn = Asum1[Asum1 < fmin]
        elif (fmin <= _EPS and fmax < (1 - _EPS)):
            Awarn = Asum1[Asum1 > fmax]
        else:
            Awarn = Asum1[(Asum1 < fmin) | (Asum1 > fmax)]
        lenAwarn = len(list(Awarn))
        if (lenAwarn):
            print('CAUTION! The next ' + str(lenAwarn) +
                  ' elements have accumulated abs freqs beyond the limits:')
            nElmPerLine = 1 + int(
                40.0 / np.mean([len(str(elm)) for elm in Awarn.index]))
            outlog = '\t\t\t'
            for i, element in enumerate(Awarn.index):
                outlog += ' ' + str(element) + ' '
                if ((i + 1) % nElmPerLine == 0):
                    outlog += '\n\t\t\t'
            if (nElmPerLine > 1):
                outlog += '\n'
            print(outlog +
                  '\nAVOID dropping these (offending) elements?')
            if(not self.__UserChooseBool()):
                A = A.drop(Awarn.index, axis='index')
                # A = self.renormalize_DF(A)
                ncols = (A.columns).size
                nrows = (A.index).size
                print('\nData now has ' + str(nrows) +
                      ' elements (rows) and ' + str(ncols) +
                      ' time registers (columns).')

        # Copy filtered absolute counts
        #  (Warning: item@time filtering will not be applied to C)
        C = A.copy()

        # Check for suspicious timming data
        Asum0 = A.sum(axis='index')
        # Warn columns which sum is far from 1 (tolerance 1e-2) or negative
        Awarn = Asum0[(np.abs(Asum0 - 1.0) > 1e-2) | (Asum0 < _EPS)]
        if (list(Awarn) != []):
            print('\nCAUTION!' +
                  ' The next accum. probabilities are outside tolerances:')
            print(Awarn)
            sabsfmean = Asum0.mean()
            sabsfstd = Asum0.std()
            sabsfsum = Asum0.sum()
            print('Absolute frequencies: mean = %.1f  std = %.1f  sum = %i' %
                  (sabsfmean, sabsfstd, sabsfsum))
            print('Drop the offending timming data?')
            if(self.__UserChooseBool()):
                A = A.drop(Awarn.index, axis='columns')
                C = C.drop(Awarn.index, axis='columns')
                ncols = (A.columns).size
                nrows = (A.index).size
                A.columns = range(1, ncols + 1)
                print('\nData now has ' + str(nrows) +
                      ' elements (rows) and ' + str(ncols) +
                      ' time registers (columns).')
            else:
                print('Now, would you like to AVOID renormalizing' +
                      ' the offending timming data?')
                if(not self.__UserChooseBool()):
                    (A, outlog, Asum0) = self.renormalize_DF(A, '')
                    print(outlog)
                    ncols = (A.columns).size
                    nrows = (A.index).size
                    print('\nData now has ' + str(nrows) +
                          ' elements (rows) and ' + str(ncols) +
                          ' time registers (columns).')
                    print('')

        # Check for item@time frequencies under the minimum allowed (fmin)
        if (fmin > _EPS):
            print('AVOID aplying the minimum frequency limit to single data?')
            if(not self.__UserChooseBool()):
                A[A < fmin] = 0
                # if (fmax<(1-_EPS)): A[A>fmax]=0
                (A, outlog, Asum0) = self.renormalize_DF(A, '')
                print(outlog)
                ncols = (A.columns).size
                nrows = (A.index).size
                print('\nData now has ' + str(nrows) +
                      ' elements (rows) and ' +
                      str(ncols) + ' time registers (columns).')
                print('')

        # The user checks the statistics and could eliminate elements
        while(True):
            smean = A.mean(axis='columns')
            sstd = A.std(axis='columns')
            ssem = sstd / np.sqrt(ncols)
            try:
                sZeroNormFreq = A.apply(
                    pd.cut, raw=False, bins=([-1, 0, 1e+20]), labels=False
                ).apply(
                    pd.value_counts, normalize=True, axis='columns'
                )[0].fillna(0)  # [0] correspond to the (-1,0] interval
            except (KeyError):
                sZeroNormFreq = pd.Series(0.0, index=smean.index)
            # slogmean = np.log10(smean)
            # slogstd = np.log10(sstd)
            # swghts = pow(smean, 2) / pow(ssem, 2)
            s = {
                '1-mean': smean,
                '2-std': sstd,
                '3-skew': A.skew(axis='columns'),
                '4-kurt': A.kurt(axis='columns'),
                '5-sem': ssem,
                '6-ZRF': sZeroNormFreq
            }
            #   '6-logmean':slogmean, '7-slogstd':slogstd,'8-wghts':swghts}
            S = pd.DataFrame(s, index=A.index)
            print('\nBasic statistics per element:')
            print(S[['1-mean', '2-std', '3-skew', '6-ZRF']])
            print('\nData now has ' + str(nrows) + ' elements (rows) and ' +
                  str(ncols) + ' time registers (columns).')
            answer = input('To remove elements, write the labels' +
                           ' comma-separated (ENTER does nothing) : ')
            sel = [x.strip() for x in answer.split(',')]
            while True:
                while ((sel != ['']) and
                        not all([x in list(A.index) for x in sel])):
                    print(answer)
                    print([x in list(A.index) for x in sel])
                    answer = input('Invalid label! Try again : ')
                    sel = [x.strip() for x in answer.split(',')]
                break

            if (sel == ['']):
                break
            else:
                A = A.drop(sel, axis='index')
                (A, outlog, Asum0) = self.renormalize_DF(A, '')
                print(outlog)
                ncols = (A.columns).size
                nrows = (A.index).size

        return(A, S, Asum0, C)


class AutomaticSession(Session):

    """Automatic session."""

    def __getstate__(self):
        """Select the object's state from parallel used attributes."""
        state = {k: self.__dict__[k] for k in (
            'fmin', 'fmax', 'respath', 'force',
            'format', 'lessplot', 'moreplot', 'core',
            'DFcols', 'verbose', 'debug'
        )}
        return state

    def __setstate__(self, state):
        """Restore instance attributes."""
        self.__dict__.update(state)

    def __init__(self, flgs, respath):
        """Class initialization."""
        # Attributes
        self.datapath = flgs.path
        try:
            os.chdir(self.datapath)
        except (OSError):
            raise Exception(
                'ERROR: Cannot change to directory "' + self.datapath + '"')
        else:
            print('\nBase working directory changed to "' +
                  self.datapath + '"')
        self.format = flgs.automatic
        self.lessplot = flgs.lessplot
        self.moreplot = flgs.moreplot
        self.fmin = flgs.fmin
        self.fmax = flgs.fmax
        self.force = flgs.force
        self.core = flgs.core
        self.verbose = flgs.verbose
        self.debug = flgs.debug
        self.respath = respath
        self.xlsxwriter = pd.ExcelWriter(self.respath + 'cmplxcruncher.xlsx')
        self.xlsxwriter_stan = pd.ExcelWriter(self.respath +
                                              'cmplxcruncher_stan.xlsx')
        self.csvwriter = None
        self.CSP_all_unW = ColorSummaryPlot(
            path=self.respath, format=self.format, name='unWeighted')
        self.CSP_all_xW = ColorSummaryPlot(
            path=self.respath, format=self.format, name='xWeighted')
        self.CSP_all_xW_stan = ColorSummaryPlot(
            xlabel=r'$\hat V\;[\sigma_V]$',
            ylabel=r'$\hat\beta\;[\sigma_\beta]$',
            path=self.respath, format=self.format, name='xWeighted_STAN'
        )
        self.DFcols = _DF_COLS
        self.DFall = pd.DataFrame(columns=self.DFcols)
        self.processedcounter = 0
        self.standardzcounter = 0
        self.ltxdoc = None
        if _useLaTeX:
            author = 'cmplxcruncher'
            if _version is not None and _verdata is not None:
                # author += ' v' + _version + ' (' + _verdata + ')'
                author += ' v' + _version
            # Document with `\maketitle` command activated
            self.ltxdoc = ltx.Document()
            self.ltxdoc.preamble.append(ltx.Command('title',
                                                'Taylor parameters results'))
            self.ltxdoc.preamble.append(ltx.Command('author',
                                                ltx.Command('texttt', author)))
            self.ltxdoc.preamble.append(ltx.Command('date', 
                                                    ltx.NoEscape(time.strftime(
                            "%a, %d %b %Y %H:%M:%S", time.localtime()))))
            self.ltxdoc.append(ltx.Command('maketitle'))
            self.ltxdoc.packages.append(ltx.Package('geometry',
                                                    options=['a4paper',
                                                             'left=10mm',
                                                             'top=10mm',
                                                             'right=10mm',
                                                             'bottom=10mm',
                                                             'noheadfoot']))
        # Main loop
        for root, dirs, files in os.walk('.'):
            # Detach Excel files and text files
            xlsfiles = []
            txtfiles = []
            if (os.path.split(self.respath)[0] == os.path.split(root)[1]):
                break  # Avoid enter in the results subdirectory
            for file in files:
                if ((file.endswith('.xls') or file.endswith('.xlsx')) and
                        not file.startswith('~') and
                        not file.startswith('_') and
                        not file.startswith('cmplxcruncher')):
                    xlsfiles.append(file)
                if (file.endswith('.txt') and not file.startswith('~') and
                        not file.startswith('_') and
                        not file.startswith('cmplxcruncher')):
                    txtfiles.append(file)
            if len(xlsfiles):
                self.excelCruncher(root, xlsfiles)
            if len(txtfiles):
                self.textCruncher(root, txtfiles)
        # Finish
        if (self.processedcounter):
            self.CSP_all_unW.end_plot()
            self.xlsxwriter.save()
            if (self.standardzcounter):
                self.xlsxwriter_stan.save()
                self.CSP_all_xW_stan.end_plot()
            self.CSP_all_xW.end_plot()
            if _useLaTeX:
                try:
                    print('\nNOTE: pdfLaTeX console output next......')
                    self.ltxdoc.generate_pdf(filepath=(self.respath +
                                                       'cmplxcruncher'),
                                             clean_tex=False)
                except:
                    print('ERROR! Call to pdfLaTeX failed!')
                else:
                    print('END: LaTeX and its PDF file successfully created!')
            print('\nEND: cmplxcruncher successfully parsed ' +
                  str(self.processedcounter) + ' files in the directory \'' +
                  self.datapath + '\'.')
            print('END: ' + str((self.DFall.index).size) +
                  ' data sets have been processed in this session.')
        else:
            print('\nEND: CC was not able to parse any file in the dir \'' +
                  self.datapath + '\'')

    def processSheet(self, *args, **kwargs):
        """Process an excel sheet

        This is usually called in parallel by ExcelCruncher
        """
        dap_lst = []  # List of tuples of cc.plots.DoAllPlots() output
        error = 3
        sheetname = args[0]
        filename = kwargs.get('filename', "processSheet-MissingFilename")
        fileNOext = kwargs.get('fileNOext',
                               os.path.splitext(os.path.split(filename)[1])[0])
        if (sheetname[0] == '_'):
            print('\n\t NOT processing sheet ' + sheetname)
        else:
            outlog = '\n\t Processing sheet ' + sheetname + '\n'
            A, outlog1 = self.File2DataFrame(filename, sheetname)
            outlog += outlog1
            A, S, Asum, C, error, outlog2 = self.automaticDataFrame(
                A, self.fmin, self.fmax, self.force)
            outlog += outlog2
            if error == -1:
                print(outlog +
                      '\t\t ABORTED! There are repeated elements!')
            elif error == 1:
                print(outlog +
                      '\t\t ABORTED! Not enough elements for analysis!')
            elif error == 2:
                print(outlog +
                      '\t\t ABORTED! Not enough time samples for analysis!')
            else:
                outlog += '\t\t Prepare data for the dataset(s)...\n'
                data_metalst = [[C, A, S, '']]  # Lst of lst of pandas DF/Ser
                if self.core:
                    data_metalst, outlog = self.split_core_tail_DF(
                        C, A, S, outlog
                    )
                outlog += '\t\t Fit data and plot figs for: '
                for C_elm, A_elm, S_elm, subset_name in data_metalst:
                    localname = fileNOext + '_' + sheetname + subset_name
                    if subset_name == '':
                        subset_name = 'ALL DATA'
                    outlog += '[' + subset_name + '] '
                    dap_tup = DoAllPlots(
                        A_elm, S_elm, C_elm.sum(axis='index'), C_elm,
                        localname, self.__getstate__()
                    )  # where Asum is calculated in place from C_elm
                    if not isinstance(dap_tup[0], pd.DataFrame):
                        error = 4
                        break
                    else:
                        dap_lst.append(dap_tup)
            print(outlog)
        return dap_lst, error

    def excelCruncher(self, root, xlsfiles):
        """Automatic process Excel files."""

        def writeResults():
            """Aux to write results to CSV and Excel files"""
            nonlocal dap_lst, DF_xls, csvwriter, CSP_xls_unW, CSP_xls_xW
            nonlocal xlsxw_correlms, xlsxw_corrtime, xlsxw_rank, fileNOext
            # Transpose dap_lst (list of lists)
            dap_lstT = [list(i) for i in zip(*dap_lst)]
            DF_xls = DF_xls.append(dap_lstT[0])
            for DF_row, correlms, corrtime, rank in dap_lst:
                nm = DF_row.index[0]
                csv_row = [
                    eval('DF_row.at[nm,\'%s\']' % key,
                         None,
                         {'DF_row': DF_row, 'nm': nm}) for key in _DF_COLS
                ]
                csv_row.insert(0, str(nm))
                csvwriter.writerow(csv_row)
                CSP_xls_unW.add_point(
                    DF_row['V'], DF_row['V_err'],
                    DF_row['beta'], DF_row['beta_err'],
                    str(nm)
                )
                CSP_xls_xW.add_point(
                    DF_row['xW_V'], DF_row['xW_V_err'],
                    DF_row['xW_beta'], DF_row['xW_beta_err'],
                    str(nm)
                )
                sheetname = str(nm).replace(fileNOext + '_', '')
                # Avoid exception in openpyxl:
                #   maximum 31 characters allowed in sheet title
                if len(sheetname) > 31:
                    sheetname = '..._' + sheetname[-26:]
                if correlms is not None:
                    correlms.to_excel(xlsxw_correlms, sheet_name=sheetname)
                if corrtime is not None:
                    corrtime.to_excel(xlsxw_corrtime, sheet_name=sheetname)
                if rank is not None:
                    rank.to_excel(xlsxw_rank, sheet_name=sheetname)

        if len(xlsfiles) is 0:
            raise Exception(
                'excelCruncher ERROR! No Excel files for crunching.')
        for file in xlsfiles:
            filename = os.path.join(root, file)
            # Process every sheet in the Excel file (except those started '_')
            print('\n\nParsing Excel file ' + filename)
            xlsx = xlrd.open_workbook(filename, on_demand=True)
            sheetnames = xlsx.sheet_names()
            if (sheetnames == []):
                print('\t WARNING: No sheets in the Excel file ' + filename)
            else:
                fileNOext = os.path.splitext(file)[0]
                DF_xls = pd.DataFrame(columns=self.DFcols)
                sheetname = ''
                with open(self.respath + fileNOext + '.csv', 'w') as csvfile:
                    csvwriter = csv.writer(csvfile, dialect='excel')
                    csvwriter.writerow(['Name'] + _DF_COLS)
                    CSP_xls_unW = ColorSummaryPlot(
                        path=self.respath,
                        name=fileNOext + '_unWeighted',
                        format=self.format
                    )
                    CSP_xls_xW = ColorSummaryPlot(
                        path=self.respath,
                        name=fileNOext + '_xWeighted',
                        format=self.format
                    )
                    # Initialize corrank data output (in Excel format)
                    xlsxw_correlms = pd.ExcelWriter(
                        self.respath + '/' + PATH_CORK +
                        fileNOext + '_correlms.xlsx'
                    )
                    xlsxw_corrtime = pd.ExcelWriter(
                        self.respath + '/' + PATH_CORK +
                        fileNOext + '_corrtime.xlsx'
                    )
                    xlsxw_rank = pd.ExcelWriter(
                        self.respath + '/' + PATH_CORK +
                        fileNOext + '_rank.xlsx'
                    )
                    kwargs = {'filename': filename, 'fileNOext': fileNOext}
                    # Enable parallelization with 'spawn' under known platforms
                    if platform.system():  # Only for known platforms
                        mpctx = mp.get_context('spawn')  # Important for OSX&Win
                        with mpctx.Pool(processes=None) as pool:
                            # processes=None as it will then use os.cpu_count()
                            async_results = [pool.apply_async(
                                call_it,
                                args=(self, 'processSheet', sheetnames[i]),
                                kwds=kwargs
                                ) for i in range(len(sheetnames))
                            ]
                            pool.close()
                            map(mp.pool.ApplyResult.wait, async_results)
                            for sheetname, (dap_lst, error) in zip(
                                    sheetnames,
                                    [r.get() for r in async_results]
                                    ):
                                if not error:
                                    writeResults()
                    else:
                        for sheetname in sheetnames:
                            (dap_lst, error) = self.processSheet(
                                sheetname, **kwargs)
                            if not error:
                                writeResults()
                    # Avoid exception in openpyxl:
                    #   maximum 31 characters allowed in sheet title
                    if len(fileNOext) > 31:
                        fileNOext = fileNOext[0:26] + '__ETC'
                    try:
                        DF_xls, DF_stn = self.standardizeByH(DF_xls)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except (StandardizationNumElmError):
                        print('\n\t CAUTION! Normalization step aborted:' +
                              ' No h_ elements!')
                        if _useLaTeX:
                            self.to_latex(fileNOext, DF_xls)
                    else:
                        DF_stn.to_excel(self.xlsxwriter_stan,
                                        sheet_name=fileNOext)
                        self.standardzcounter += 1
                        if _useLaTeX:
                            self.to_latex(fileNOext, DF_xls, DF_stn)
                    print('\nSaving results for ' + filename + '...')
                    DF_xls.to_excel(self.xlsxwriter, sheet_name=fileNOext)
                    CSP_xls_unW.end_plot()
                    CSP_xls_xW.end_plot()
                    # Save corrank data output (in Excel format)
                    if xlsxw_correlms.sheets:
                        xlsxw_correlms.save()
                    if xlsxw_corrtime.sheets:
                        xlsxw_corrtime.save()
                    if xlsxw_rank.sheets:
                        xlsxw_rank.save()
                    # End of with open() as csvfile
                self.DFall = self.DFall.append(DF_xls)
                self.CSP_all_unW.add_point(
                    DF_xls['V'], DF_xls['V_err'], DF_xls['beta'],
                    DF_xls['beta_err'], fileNOext
                )
                self.CSP_all_xW.add_point(
                    DF_xls['xW_V'], DF_xls['xW_V_err'], DF_xls['xW_beta'],
                    DF_xls['xW_beta_err'], fileNOext
                )
                try:
                    self.CSP_all_xW_stan.add_point(
                        DF_xls['xW_V_stan'], DF_xls['xW_V_err_stan'],
                        DF_xls['xW_beta_stan'], DF_xls['xW_beta_err_stan'],
                        fileNOext
                    )
                except:
                    pass
                self.processedcounter += 1

    def processTxt(self, *args, **kwargs):
        """Process a txt file (to be usually called in parallel!)."""
        txtfile = args[0]
        root = kwargs.get('root', "processTxt-MissingRoot")
        txtfileNOext = os.path.splitext(txtfile)[0]
        txtfilename = os.path.join(root, txtfile)
        error = 3
        dap_lst = []
        if (txtfile[0] == '_'):
            print('\n\n NOT processing txt file ' + txtfilename)
        else:
            outlog = '\n\n Parsing TXT file ' + txtfilename + '\n'
            A, outlog1 = self.File2DataFrame(txtfilename, '')
            outlog += outlog1
            A, S, Asum, C, error, outlog2 = self.automaticDataFrame(
                A, self.fmin, self.fmax, self.force)
            outlog += outlog2
            if error == -1:
                print(outlog +
                      '\t\t ABORTED! There are repeated elements!')
            elif error == 1:
                print(outlog +
                      '\t\t ABORTED! Not enough elements for analysis!')
            elif error == 2:
                print(outlog +
                      '\t\t ABORTED! Not enough time samples for analysis!')
            else:
                outlog += '\t\t Prepare data for the dataset(s)...\n'
                data_metalst = [[C, A, S, '']]  # Lst of lst of DataFrms/Series
                if self.core:
                    data_metalst, outlog = self.split_core_tail_DF(
                        C, A, S, outlog
                    )
                outlog += '\t\t Fit data and plot figs for: '
                for C_elm, A_elm, S_elm, subset_name in data_metalst:
                    localname = txtfileNOext + '_' + subset_name
                    if subset_name == '':
                        subset_name = 'ALL DATA'
                    outlog += '[' + subset_name + '] '
                    dap_tup = DoAllPlots(
                        A_elm, S_elm,
                        C_elm.sum(axis='index'), C_elm,
                        self.respath, localname,
                        self.format, self.lessplot,
                        self.moreplot, self.DFcols
                    )  # where Asum is calculated in place from C_elm
                    if not isinstance(dap_tup[0], pd.DataFrame):
                        error = 4
                        break
                    else:
                        dap_lst.append(dap_tup)
                print(outlog)
        return dap_lst, error

    def textCruncher(self, root, txtfiles):
        """Automatic process txt files."""

        def writeResults():
            """Aux to plot CSP and write results to CSV and Excel files"""
            nonlocal xlsxw_correlms, xlsxw_corrtime, xlsxw_rank, dap_lst
            txtfileNOext = os.path.splitext(txtfile)[0]
            # Avoid exception in openpyxl:
            #   maximum 31 characters allowed in sheet title
            if len(txtfileNOext) > 31:
                txtfileNOext = txtfileNOext[0:26] + '__ETC'
            # Transpose dap_lst (list of lists)
            dap_lstT = [list(i) for i in zip(*dap_lst)]
            DF_txt = pd.concat(dap_lstT[0])
            self.DFall = self.DFall.append(DF_txt)
            DF_txt.to_excel(self.xlsxwriter, sheet_name=txtfileNOext)
            if self.core:
                CSP_one_unW = ColorSummaryPlot(
                    path=self.respath, format=self.format,
                    name=txtfileNOext + '_unWeighted'
                )
                CSP_one_xW = ColorSummaryPlot(
                    path=self.respath, format=self.format,
                    name=txtfileNOext + '_xWeighted'
                )
                for DF_row, correlms, corrtime, rank in dap_lst:
                    name = str(DF_row.index[0])
                    CSP_one_unW.add_point(
                        DF_row['V'], DF_row['V_err'],
                        DF_row['beta'], DF_row['beta_err'],
                        name
                    )
                    CSP_one_xW.add_point(
                        DF_row['xW_V'], DF_row['xW_V_err'],
                        DF_row['xW_beta'], DF_row['xW_beta_err'],
                        name
                    )
                    # Avoid exception in openpyxl:
                    #   maximum 31 characters allowed in sheet title
                    if len(name) > 31:
                        name = '..._' + name[-26:]
                    if correlms is not None:
                        correlms.to_excel(xlsxw_correlms, sheet_name=name)
                    if corrtime is not None:
                        corrtime.to_excel(xlsxw_corrtime, sheet_name=name)
                    if rank is not None:
                        rank.to_excel(xlsxw_rank, sheet_name=name)
                CSP_one_unW.end_plot()
                CSP_one_xW.end_plot()
            else:
                try:
                    [(DF_row, correlms, corrtime, rank)] = dap_lst
                except:
                    print('ERROR! Unable to retrieve texfile corrank data!')
                else:
                    if correlms is not None:
                        correlms.to_excel(xlsxw_correlms,
                                          sheet_name=txtfileNOext)
                    if corrtime is not None:
                        corrtime.to_excel(xlsxw_corrtime,
                                          sheet_name=txtfileNOext)
                    if rank is not None:
                        rank.to_excel(xlsxw_rank,
                                      sheet_name=txtfileNOext)
            self.CSP_all_unW.add_point(
                DF_txt['V'], DF_txt['V_err'],
                DF_txt['beta'], DF_txt['beta_err'],
                txtfileNOext
            )
            self.CSP_all_xW.add_point(
                DF_txt['xW_V'], DF_txt['xW_V_err'],
                DF_txt['xW_beta'], DF_txt['xW_beta_err'],
                txtfileNOext
            )

        if len(txtfiles) is 0:
            raise Exception('textCruncher ERROR! No TXT files for crunching.')
        kwargs = {'root': root}
        # Initialize corrank data output (in Excel format)
        xlsxw_correlms = pd.ExcelWriter(
            self.respath + '/' + PATH_CORK + 'txtfiles_correlms.xlsx')
        xlsxw_corrtime = pd.ExcelWriter(
            self.respath + '/' + PATH_CORK + 'txtfiles_corrtime.xlsx')
        xlsxw_rank = pd.ExcelWriter(
            self.respath + '/' + PATH_CORK + 'txtfiles_rank.xlsx')
        # Enable parallelization with 'spawn' under known platforms
        if platform.system():  # Only for known platforms
            mpctx = mp.get_context('spawn')  # Important for OS X and Win
            with mpctx.Pool(processes=None) as pool:  # Use os.cpu_count()
                async_results = [pool.apply_async(
                    call_it,
                    args=(self, 'processTxt', txtfiles[i]), kwds=kwargs
                ) for i in range(len(txtfiles))]
                pool.close()
                map(mp.pool.ApplyResult.wait, async_results)
                for txtfile, (dap_lst, error) in zip(
                        txtfiles, [r.get() for r in async_results]):
                    if not error:
                        writeResults()
                    self.processedcounter += 1
        else:
            for txtfile in txtfiles:
                (dap_lst, error) = self.processTxt(txtfile, **kwargs)
                if not error:
                    writeResults()
                self.processedcounter += 1
        # Save corrank data output (in Excel format)
        if xlsxw_correlms.sheets:
            xlsxw_correlms.save()
        if xlsxw_corrtime.sheets:
            xlsxw_corrtime.save()
        if xlsxw_rank.sheets:
            xlsxw_rank.save()

    def automaticDataFrame(self, A, fmin, fmax, force):
        """Automatic process pandas DataFrame."""
        error = 0  # Error condition
        outlog = ''  # Log string
        ncols = (A.columns).size
        nrows = (A.index).size
        outlog = (outlog + '\t\t Just read ' + str(nrows) +
                  ' elements (rows) and ' + str(ncols) +
                  ' time registers (columns)...\n')

        # Check for repeated elements (rows) and RETURN if so
        #  (alternate exit point in the function!)
        if (len(A.index.values) != len(set(A.index.values))):
            outlog = (outlog + '\t\t The next elements are repeated' +
                      ' the shown number of times:\n\t\t\t' +
                      str([(x, y) for x, y in Counter(A.index.values).items()
                           if y > 1]) + '\n')
            if(not force):
                error = -1
                return(A, [], [], [], error, outlog)
            outlog = (outlog + '\t\t WARNING! You decided to go ahead,' +
                      ' but be aware of inconsistencies! \n')

        # Check for elements with normalized accum. freqs beyond desired limits
        Asum1 = A.sum(axis='columns')
        Asum1 = Asum1.div(Asum1.sum())  # Normalize sum
        # Next IF is the legacy case: check only elements with null frequencies
        if (fmin <= _EPS and fmax >= (1 - _EPS)):
            Awarn = Asum1[Asum1 < _EPS]
        elif (fmin > _EPS and fmax >= (1 - _EPS)):
            Awarn = Asum1[Asum1 < fmin]
        elif (fmin <= _EPS and fmax < (1 - _EPS)):
            Awarn = Asum1[Asum1 > fmax]
        else:
            Awarn = Asum1[(Asum1 < fmin) | (Asum1 > fmax)]
        if list(Awarn):
            outlog = (
                outlog + '\t\t WARNING! Eliminating ' + str(len(Awarn)) +
                ' elements with acc. abs freqs beyond the limits...\n'
            )
            if self.verbose:
                outlog += '\t\t\t'
                nElmPerLine = 1 + int(
                    40.0 / np.mean([len(str(elm)) for elm in Awarn.index])
                )
                for i, element in enumerate(Awarn.index):
                    outlog += ' ' + str(element) + ' '
                    if ((i + 1) % nElmPerLine == 0):
                        outlog += '\n\t\t\t'
                if (nElmPerLine > 1):
                    outlog += '\n'
            A = A.drop(Awarn.index, axis='index')

        # Copy filtered absolute counts
        # (Warning: item@time filtering will not be applied to C)
        C = A.copy()

        # Normalize to get relative frequencies
        (A, outlog, Asum0) = self.renormalize_DF(A, outlog)

        # Check for item@time frequencies under the minimum allowed (fmin)
        if (fmin > _EPS and (A.index).size > 0):
            A[A < fmin] = 0
            # if (fmax<(1-_EPS)): A[A>fmax]=0
            (A, outlog, Asum) = self.renormalize_DF(A, outlog)

        # Compute basic statistics in DataFrame S
        ncols = (A.columns).size
        nrows = (A.index).size
        if (nrows < 5):
            error = 1  # Not enough elements for analysis!
            S = []
        elif (ncols < 2):
            error = 2  # Not enough time samples for analysis!
            S = []
        else:
            outlog = (outlog + '\t\t Processing ' + str(nrows) +
                      ' elements (rows) and ' + str(ncols) +
                      ' time registers (columns)...\n')
            smean = A.mean(axis='columns')
            sstd = A.std(axis='columns')
            ssem = sstd / np.sqrt(ncols)
            try:
                sZeroNormFreq = A.apply(
                    pd.cut, raw=False, bins=([-1, 0, 1e+20]), labels=False
                ).apply(
                    pd.value_counts, normalize=True, axis='columns'
                )[0].fillna(0)  # [0] correspond to the (-1,0] interval
            except (KeyError):
                sZeroNormFreq = pd.Series(0.0, index=smean.index)
            s = {
                '1-mean': smean,
                '2-std': sstd,
                '3-skew': A.skew(axis='columns'),
                '4-kurt': A.kurt(axis='columns'),
                '5-sem': ssem,
                '6-ZRF': sZeroNormFreq
            }
            S = pd.DataFrame(s, index=A.index)

        return(A, S, Asum0, C, error, outlog)

    def standardizeByH(self, DFsrc):
        """Get standardization values for the elements."""
        # Select h_ elements as basin for the standardization
        DFh = DFsrc.loc[
            DFsrc.index.map(lambda name: "h_" in name),
            ['xW_V', 'xW_V_err', 'xW_beta', 'xW_beta_err']
        ]
        numh = (DFh.index).size
        if numh > 1:
            print('\n\t Standardization step: by ' + str(numh) +
                  ' h_ elements.' + ' Stan. matrix:')
        else:
            raise StandardizationNumElmError(self, sys._getframe())
        var_lst = ['xW_V', 'xW_beta']
        DFn = pd.DataFrame(
            {'mean_w': [None, None], 'stdv_w': [None, None]},
            index=pd.Index(var_lst)
        )
        index = pd.Index(var_lst)
        for var in var_lst:
            # h_ elements auxiliar computations
            DFh['1/' + var + '_err^2'] = 1.0 / DFh[var + '_err'] ** 2
            sem2 = 1 / DFh['1/' + var + '_err^2'].sum()
            DFh[var + '_weights'] = DFh['1/' + var + '_err^2'] * sem2
            DFh[var + '_w'] = DFh[var] * DFh[var + '_weights']
            # Calculate standardization magnitudes: weighted mean and std dev
            DFn.loc[var, 'mean_w'] = DFh[var + '_w'].sum()
            DFn.loc[var, 'stdv_w'] = np.sqrt(
                (DFh[var + '_weights'] * (
                    DFh[var] - DFn.loc[var, 'mean_w']
                ) ** 2).sum() /
                (1 - (DFh[var + '_weights'] ** 2).sum())
            )
            # Calculate other magnitudes only for output comparison
            DFn.loc[var, '__sigm'] = np.sqrt(sem2 * numh)
            DFn.loc[var, '__disp'] = np.sqrt(
                ((DFh[var] - DFn.loc[var, 'mean_w']) ** 2).sum() / numh
            )
            DFn.loc[var, '_stdv'] = np.sqrt(
                DFn.loc[var, '__sigm'] ** 2 + DFn.loc[var, '__disp'] ** 2
            )
            # Normalize all the elements by the standardization magnitudes
            DFsrc[var + '_stan'] = (    # Standardization value
                (DFsrc[var] - DFn.loc[var, 'mean_w']) / DFn.loc[var, 'stdv_w']
            )
            DFsrc[var + '_err_stan'] = (    # Standardized error
                DFsrc[var + '_err'] / DFn.loc[var, 'stdv_w']
            )
        print(DFn)
        return(DFsrc, DFn)

    def to_latex(self, secname, DFsrc, DFn=None):
            """Export to LaTeX tables the main results."""
            _f = '{:L}'  # LaTeX output format in uncertainties library
            _fM = '$' + _f + '$'  # As before, but for LaTeX math mode
            _fR2 = '${:1.3f}$'
            self.ltxdoc.append(
                    ltx.NoEscape(r'%% Section for the dataset ' + secname)
            )
            if DFn is not None:
                # Create table with standardization results
                with self.ltxdoc.create(ltx.Section(
                        ltx.utils.escape_latex(secname))
                        ):
                    self.ltxdoc.append(
                            ltx.Command('begin', 'table', 'h',
                                        extra_arguments='')
                    )
                    self.ltxdoc.append(ltx.Command('centering'))
                    with self.ltxdoc.create(ltx.Tabular('ccccccc')) as tbl:
                        tbl.add_hline()
                        tbl.add_row(('Metadata', 'V',
                                     ltx.NoEscape(r'$\beta$'),
                                     ltx.NoEscape(r'$\bar{R}^2$'), '',
                                     ltx.NoEscape(r'V$_{st}$'),
                                     ltx.NoEscape(r'$\beta_{st}$')))
                        tbl.add_hline()
                        for bool in [True, False]:
                            for index, row in DFsrc.iterrows():
                                if ("h_" in index) is bool:
                                    v = _fM.format(
                                        ufloat(row['xW_V'], row['xW_V_err'])
                                        )
                                    beta = _fM.format(
                                        ufloat(row['xW_beta'],
                                               row['xW_beta_err'])
                                        )
                                    R2 = _fR2.format(row['xW_R^2'])
                                    v_stan = _fM.format(
                                        ufloat(row['xW_V_stan'],
                                               row['xW_V_err_stan'])
                                        )
                                    beta_stan = _fM.format(
                                        ufloat(row['xW_beta_stan'],
                                               row['xW_beta_err_stan'])
                                        )
                                    metadata = str(index).replace(secname +
                                                                  '_', '')
                                    tbl.add_row((
                                        metadata,
                                        ltx.NoEscape(v), ltx.NoEscape(beta),
                                        ltx.NoEscape(R2), '', 
                                        ltx.NoEscape(v_stan),
                                        ltx.NoEscape(beta_stan)
                                        ))
                            tbl.add_hline()
                        tbl.add_hline()
                    xW_V_mean_w = _f.format(ufloat(DFn.loc['xW_V', 'mean_w'],
                                                   DFn.loc['xW_V', 'stdv_w']))
                    xW_beta_mean_w = _f.format(
                        ufloat(DFn.loc['xW_beta', 'mean_w'],
                               DFn.loc['xW_beta', 'stdv_w'])
                        )
                    caption = ltx.NoEscape('Taylor parameters for the dataset ' +
                               ltx.utils.escape_latex(secname) +
                               r'. The healthy population is described by ' +
                               r'$\bar{V} = ' + xW_V_mean_w +
                               r', \bar{\beta} = ' +
                               xW_beta_mean_w + r'$.'
                               )
                    self.ltxdoc.append(ltx.Command('caption', caption))
                    self.ltxdoc.append(ltx.Command('end', 'table'))
            else:
                # Create table without standardization results
                with self.ltxdoc.create(ltx.Section(
                        ltx.utils.escape_latex(secname))
                        ):
                    self.ltxdoc.append(
                            ltx.Command('begin', 'table', 'h',
                                        extra_arguments='')
                    )
                    self.ltxdoc.append(ltx.Command('centering'))
                    with self.ltxdoc.create(ltx.Tabular('cccc')) as tbl:
                        tbl.add_hline()
                        tbl.add_row(('Metadata', 'V',
                                       ltx.NoEscape(r'$\beta$'),
                                       ltx.NoEscape(r'$\bar{R}^2$')))
                        tbl.add_hline()
                        for index, row in DFsrc.iterrows():
                            v = _fM.format(ufloat(row['xW_V'],
                                                  row['xW_V_err']))
                            beta = _fM.format(ufloat(row['xW_beta'],
                                                     row['xW_beta_err']))
                            R2 = _fR2.format(row['xW_R^2'])
                            metadata = str(index).replace(secname + '_', '')
                            tbl.add_row((ltx.utils.escape_latex(metadata),
                                           v, beta, R2))
                        tbl.add_hline()
                        tbl.add_hline()
                    caption = ltx.NoEscape('Taylor parameters for the dataset ' +
                               ltx.utils.escape_latex(secname) + '.')
                    self.ltxdoc.append(ltx.Command('caption', caption))
                    self.ltxdoc.append(ltx.Command('end', 'table'))
            self.ltxdoc.append(ltx.Command('clearpage'))
