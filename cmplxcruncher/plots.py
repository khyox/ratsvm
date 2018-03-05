"""
This module provides the plot classes of the package.
"""

# Packages
import os
import itertools
import pandas as pd
import numpy as np
import concurrent.futures
from scipy import optimize, stats
# import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
# plt.switch_backend('GTK3Cairo')
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import matplotlib.cm as cmx
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Ellipse
from cmplxcruncher.config import *
from cmplxcruncher.models import PowerFitModel, CovMatrixError

# Constants
_EPS = np.finfo(np.double).eps

# Define "module static" variable devflag
devflag = False


def enableDevMode():
    global devflag
    devflag = True


def disableDevMode():
    global devflag
    devflag = False


def getDevMode():
    global devflag
    return (devflag)


class CCplot(object):
    """Base CC plot through Matplotlib.pyplot."""

    def __init__(self,
                 xlabel, ylabel,
                 xscale='lin', yscale='lin',
                 path='.', name='', plot='',
                 format='png', title='',
                 skipfigure=False
                 ):
        # Initialize data members
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.path = path
        self.name = name
        self.plot = plot
        self.format = format
        self.title = title
        # Create the figure (otherwise should be created later)
        if not skipfigure:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.ax.grid()
            self.ax.set_xmargin(0.1)
            self.ax.set_ymargin(0.1)

    def end_plot(self, plot=None):
        """Plot to screen or save the figure."""
        if plot:
            self.plot = plot
        # Set scales
        if not isinstance(self.ax, np.ndarray):
            if self.xscale == 'lin':
                self.ax.set_xlabel(self.xlabel + ' (lineal scale)')
            elif self.xscale == 'log':
                self.ax.set_xlabel(self.xlabel + ' (log scale)')
                self.ax.set_xscale('log', nonposx='clip')
            elif self.xscale == 'hist2D':
                self.ax.grid(False)
                self.ax.set_xlabel(self.xlabel, fontsize=20)
            elif self.xscale == 'nogrid':
                self.ax.grid(False)
                self.ax.set_xlabel(self.xlabel)
            elif self.xscale == 'none':
                self.ax.get_xaxis().set_ticks([])
                plt.setp(self.ax.get_xticklines(), visible=False)
            else:
                raise Exception('ERROR! CCplot.end_plot(): invalid xscale \'' +
                                str(self.yscale) +
                                '\' (options: \'lin\', \'log\', \'none\')')
            if self.yscale == 'lin':
                self.ax.set_ylabel(self.ylabel + ' (lineal scale)')
            elif self.yscale == 'log':
                self.ax.set_ylabel(self.ylabel + ' (log scale)')
                self.ax.set_yscale('log', nonposy='clip')
            elif self.yscale == 'hist2D':
                self.ax.grid(False)
                self.ax.set_ylabel(self.ylabel, fontsize=20)
            elif self.yscale == 'nogrid':
                self.ax.grid(False)
                self.ax.set_ylabel(self.ylabel)
            elif self.yscale == 'none':
                self.ax.get_yaxis().set_ticks([])
                plt.setp(self.ax.get_yticklines(), visible=False)
            else:
                raise Exception('ERROR! CCplot.end_plot(): invalid yscale \'' +
                                str(self.yscale) +
                                '\' (options: \'lin\', \'log\', \'none\')')
        # Try to set title
        if self.title:
            try:
                self.ax.set_title(self.title)
            except(AttributeError):
                try:
                    self.fig.suptitle(self.title, size=18)
                except:
                    pass
        # Plot or save
        if (self.format == ''):
            plt.show()
        else:
            name = self.path + self.name + self.plot
            if self.format == 'png':  # Raster image
                pngname = name + '.png'
                self.fig.savefig(pngname, dpi=200)
            else:  # Vectorial image
                imagename = name + '.' + self.format
                self.fig.savefig(imagename)
        # Close the plot to release memory
        plt.close()


class ErrorPlot(CCplot):
    """Plot an error message."""

    # Initialization
    def __init__(self, xlabel='', ylabel='', xscale='lin', yscale='lin',
                 path='.', name='', plot='', format='png', title='', msg=''):
        # Call superclass initialization
        super(ErrorPlot, self).__init__(
            xlabel, ylabel, xscale, yscale,
            path, name, plot, format, title
        )
        # Do ALL the staff
        if msg == '':
            msg = "Sorry, error plotting this figure\nPlease review your data"
        else:
            msg = str(msg)
        xpos = ypos = 0.5
        if xscale == 'log':
            xpos = 1.0e-4
        if yscale == 'log':
            ypos = 1.0e-4
        plt.text(
            xpos, ypos, msg, size=20,
            ha="center", va="center",
            bbox=dict(
                boxstyle="round", ec=(1., 0.4, 0.4),
                fc=(1., 0.7, 0.7)
            )
        )
        super(ErrorPlot, self).end_plot()


class LinearPlot(CCplot):
    """Plot a scattered linear plot."""

    def __init__(self,
                 xdata, ydata,
                 xlabel='', ylabel='',
                 path='.', name='', format='png'
                 ):
        # Call superclass initialization
        super(LinearPlot, self).__init__(
            xlabel, ylabel, xscale='lin', yscale='lin',
            path=path, name=name, format=format
        )
        # Plot the graphs
        plt.plot(xdata, ydata, 'ro')
        self.title = self.name + ': Linear Plot'
        super(LinearPlot, self).end_plot(plot='_' + ylabel + 'VS' + xlabel)


class SemilogxPlot(CCplot):
    """Plot a scattered semilogx plot."""

    def __init__(self,
                 xdata, ydata,
                 xlabel='', ylabel='',
                 path='.', name='', format='png'
                 ):
        # Call superclass initialization
        super(SemilogxPlot, self).__init__(
            xlabel, ylabel, xscale='log', yscale='lin',
            path=path, name=name, format=format
        )
        # Plot the graphs
        plt.semilogx(xdata, ydata, 'bo')
        self.title = self.name + ': Semilogx Plot'
        super(SemilogxPlot, self).end_plot(plot='_' + ylabel + 'VS' + xlabel)


class Hist2Dplot(CCplot):
    """Do a 2D histogram of the difference of data with its mean."""

    def __init__(self,
                 A, S, xlabel=r'$\log_{10}\left(\bar{x}\right)$',
                 ylabel=r'$\Delta\bar x$', path='.', name='', format='png'
                 ):
        # Call superclass initialization
        super(Hist2Dplot, self).__init__(
            xlabel, ylabel, xscale='hist2D', yscale='hist2D',
            path=path, name=name, plot='_hist2D', format=format
        )
        # Do the specific staff
        x_2D = []
        y_2D = []
        for col in A.columns:
            x_2D = np.concatenate([x_2D, np.log10(S['1-mean'])])
            y_2D = np.concatenate([y_2D, (S['1-mean'] - A[col])])
        # Plot the graphs
        plt.hist2d(x_2D, y_2D, bins=[10, 20], norm=colors.LogNorm())
        plt.colorbar()
        self.title = self.name + ': 2D deviation plot'
        super(Hist2Dplot, self).end_plot()


class CorrPlot(CCplot):
    """
    Hinton Diagram of the Pearson-correlation matrix of DF.

    axis -- string, options: 'elements' correlation among elements
                             'times' correlation among time points
    maxelm -- Truncate the elements corr table to this number of elements
    maxtms -- Not to annotate the correlations for more times than this number
    maxweight -- Select a weight or let unchanged to be inferred
    """

    # Aux method to draw a square-shaped blob with the given
    #   area (< 1) at the given coordinates
    def _blob(self, x, y, area, colour):
        hs = np.sqrt(area) / 2
        xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
        plt.fill(xcorners, ycorners, colour, edgecolor=colour)

    def __init__(self,
                 indf, axis,
                 maxelm=15, maxtms=20, maxweight=None,
                 path='.', name='', format='png'
                 ):
        # Call superclass initialization
        super(CorrPlot, self).__init__(
            '', '', xscale='nogrid', yscale='nogrid',
            path=path, name=name, plot='_UNDEF', format=format
        )
        # Do the specific staff
        self.corrdf = None
        ticks = []
        xticks = []
        yticks = []
        # Create the Hinton Diagram
        if (axis == 'elements'):
            self.title = self.name + ': elements corr matrix'
            # Get a copy of the caller-namescope DataFrame and
            #   make auxiliary DataFrame `df` point to it
            df = indf.copy()
            # Calc column of accum elements, rank, reorder by, and del it
            df['SUM'] = df.sum(axis='columns')
            df = df.sort_values(by='SUM', ascending=False)
            df.drop('SUM', axis='columns', inplace=True)
            self.corrdf = df.T.corr()
            # Lets `df` point to maxelm-truncated view of the correlation DF
            try:
                df = self.corrdf.iloc[:maxelm, :maxelm]
            except(IndexError):
                df = self.corrdf
            W = df.values
            xticks = ['elm' + str(x + 1) for x in range((df.index).size)]
            yticks = [x + ':' + y for x, y in zip(xticks, df.index.values)]
            if not format == '':
                try:
                    ticks = [
                        (sel[1].strip() + '_*_' + sel[-1].strip()) for sel in [
                            elm.split('_') for elm in df.index.values
                            ]
                        ]
                except:
                    try:
                        ticks = [
                            (sel[1].strip() + ';*;' + sel[-1].strip())
                            for sel in [
                                elm.split(';') for elm in df.index.values
                                ]
                            ]
                    except:
                        pass
                    else:
                        yticks = [x + ':' + y for x, y in zip(xticks, ticks)]
                else:
                    yticks = [x + ':' + y for x, y in zip(xticks, ticks)]
        elif (axis == 'times'):
            self.corrdf = indf.corr()
            W = self.corrdf.values
            self.title = self.name + ': times corr matrix'
            ticks = self.corrdf.columns.values
        else:
            raise Exception('ERROR! In function DFcorrPlot: invalid axis "' +
                            str(axis) + '" (options: "elements", "times")')
        height, width = W.shape
        if not maxweight:
            maxweight = 2 ** np.ceil(np.log(np.max(np.abs(W))) / np.log(2))
        plt.fill(
            np.array([0, width, width, 0]),
            np.array([0, 0, height, height]),
            'gray'
        )
        plt.axis('scaled')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.setp(self.ax.get_xticklines(), visible=False)
        plt.setp(self.ax.get_yticklines(), visible=False)
        for x in range(width):
            for y in range(height):
                _x = x + 1
                _y = y + 1
                w = W[y, x]
                if w > 0:
                    self._blob(_x - 0.5,
                               height - _y + 0.5,
                               min(1, w / maxweight),
                               'white')
                elif w < 0:
                    self._blob(_x - 0.5,
                               height - _y + 0.5,
                               min(1, -w / maxweight),
                               'black')
        if axis == 'elements':
            # Annotate the correlation for every element
            for x in range(width):
                for y in range(height):
                    self.ax.annotate(
                        r'%1.2f' % W[x][y],
                        xy=(x + 0.5, height - (y + 0.5)),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size='xx-small',
                        color=('white' if W[x][y] < 0 else 'black')
                    )
        elif axis == 'times':
            if len(ticks) > 50:
                pass  # Too many times for annotating
            elif len(ticks) > maxtms:
                # Annotate only the diagonal
                #  for 'time' with more than maxtms times but less than 51
                diagsize = np.minimum(width, height)
                for t in range(diagsize):
                    self.ax.annotate(
                        str(t + 1),
                        xy=(t + 0.5, height - (t + 0.5)),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size='xx-small'
                    )
            else:
                # Annotate the correlation for every element
                for x in range(width):
                    for y in range(height):
                        self.ax.annotate(
                            r'%1.2f' % W[x][y],
                            xy=(x + 0.5, height - (y + 0.5)),
                            horizontalalignment='center',
                            verticalalignment='center',
                            size='xx-small',
                            color=('white' if W[x][y] < 0 else 'black')
                        )
        else:
            raise Exception(
                'ERROR! In function DFcorrPlot: invalid axis "' +
                str(axis) + '" (options: "elements", "times")')

        # Set ticks and output the adequate plot or the image
        if axis == 'elements':
            plt.yticks(
                np.arange(len(yticks) - 0.5, 0, -1),
                yticks, size='x-small'
            )
            plt.xticks(
                np.arange(0.5, len(xticks) + 0.5, 1),
                xticks, size='x-small', rotation=60
            )
            self.fig.set_tight_layout(True)
            super(CorrPlot, self).end_plot(plot='_ElementsCorr')
        elif axis == 'times':
            if len(ticks) <= maxtms:
                plt.yticks(
                    np.arange(len(ticks) - 0.5, 0, -1),
                    ticks, size='x-small'
                )
                plt.xticks(
                    np.arange(0.5, len(ticks) + 0.5, 1),
                    ticks, size='x-small', rotation=60
                )
            else:
                # Display reversed ticks on yaxis
                locs, labels = plt.yticks()
                plt.yticks([len(ticks)-loc for loc in locs],
                           [str(int(lab)) for lab in locs])
            super(CorrPlot, self).end_plot(plot='_TimesCorr')
        else:
            raise Exception('ERROR! In function DFcorrPlot: invalid axis "' +
                            str(axis) + '" (options: "elements", "times")')


class RankPlot(CCplot):
    """
    Rank matrix plot with Rank Stability Index of DataFrame.

    maxelm -- Truncate the printed table to this number of elements
    """

    def __init__(self, C, maxelm=50, path='.', name='', format='png'):
        # Call superclass initialization but do custom figure init.
        super(RankPlot, self).__init__(
            '', '', xscale='nogrid', yscale='nogrid',
            path=path, name=name, plot='_Rank', format=format, skipfigure=True,
            title=name + ': rank matrix & stability'
        )
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, squeeze=False,
                                         sharey='row', sharex='col',
                                         gridspec_kw={'width_ratios': [15, 1],
                                                      'height_ratios': [4, 1]},
                                         figsize=(12, 12)
                                         )
        self.fig.set_tight_layout({'rect': [0.0, 0.0, 1.0, 0.95],
                                   'w_pad': 0.1, 'h_pad': 0.1})
        axrank = self.ax[0, 0]
        axRSI = self.ax[0, 1]
        axrvar = self.ax[1, 0]
        self.fig.delaxes(self.ax[1, 1])

        # Do the specific calculations
        self.rankdf = None
        # Avoid mod C: Get in R the address of a copy of the DF pointed by C
        R = C.copy()
        # Calculate column of accumulated elements, rank, reorder by, and del
        R['Stab'] = R.sum(axis='columns')
        R = R.rank(ascending=False, method='first').astype('int')
        R = R.sort_values(by='Stab', ascending=False)
        R.drop('Stab', axis='columns', inplace=True)
        # Calculate difference matrix in the time axis
        D = R.T.diff().T.iloc[:, 1:]
        # Calculate normalized averaged deviation of differences
        Ddev = []
        try:
            Dmaxelem = D.iloc[-maxelm:]
        except IndexError:
            Ddev = D.abs().divide(len(D.index)).mean().values
        else:
            Ddev = Dmaxelem.abs().divide(len(Dmaxelem.index)).mean().values
        # Truncate R DataFrame to maxelm (D remains unchanged)
        try:
            R = R.iloc[-maxelm:]
        except IndexError:
            pass
        # Calculate average of abs(r_i/rmean - 1) with r_i rank of taxon i.
        Smaxelm = R.divide(R.mean(axis='columns'), axis='index').subtract(
            1).abs().mean().values
        W = R.values
        height, width = W.shape

        # Absolute rank subplot
        xticks = R.columns.values
        yticks = R.index.values[::-1]
        # In file plots, if the labels are in QIIME fomat, shorten them
        if not format == '':
            try:
                ticks = [
                    (sel[1].strip() + '_*_' + sel[-1].strip()) for sel in [
                        elm.split('_') for elm in (R.index.values[::-1])
                        ]
                    ]
            except:
                try:
                    ticks = [
                        (sel[1].strip() + ';*;' + sel[-1].strip()) for sel in [
                            elm.split(';') for elm in (R.index.values[::-1])
                            ]
                        ]
                except:
                    pass
                else:
                    yticks = ticks
            else:
                yticks = ticks
        pcm = axrank.pcolormesh(W, cmap=plt.cm.inferno_r, shading='flat')
        axrank.set_xlim(0, width)
        axrank.set_ylim(0, height)
        plt.setp(axrank.get_xticklines(), visible=False)
        plt.setp(axrank.get_yticklines(), visible=False)
        # Annotate the rank matrix, if not so many samples
        if len(xticks) < maxelm:
            rankmax = np.max(W)
            for x in range(width):
                for y in range(height):
                    axrank.annotate(
                        str(W[y][x]), xy=(x + 0.5, y + 0.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size='xx-small',
                        color=('white' if W[y][x] > rankmax/2 else 'black')
                    )
        # Set ticks and output the adequate plot for the image
        axrank.yaxis.set_ticks(np.arange(len(yticks) - 0.5, 0, -1))
        axrank.yaxis.set_ticklabels(yticks, size='small')

        # Averaged rank variability plot
        smaxplt = axrvar.plot(np.arange(0.5, len(Smaxelm), 1),
                              Smaxelm, color='red', lw=2.5,
                              label='RV (' + str(height) + ' most abundant)')
        # axrvar.vlines(np.arange(0.5, len(Smaxelm), 1),
        #              0, Smaxelm, color='red')
        axrvar.set_ylim(ymin=0, ymax=1.8)
        axrvar.set_xlim(xmin=0, xmax=width)
        # Avoid explicit time labels if too many
        if len(xticks) < maxelm:
            axrvar.xaxis.set_ticks(np.arange(0.5, len(xticks) + 0.5, 1))
            axrvar.xaxis.set_ticklabels(xticks, size='xx-small', rotation=60)
        axrvar.set_xlabel('Time samples')
        axrvar.set_ylabel('Rank Variability', color='red')
        plt.setp(axrvar.yaxis.get_ticklabels(), size='small')
        # Make the y-axis label and tick labels match the line color.
        for tl in axrvar.get_yticklabels():
            tl.set_color('red')
        axrvar_ = axrvar.twinx()
        Ddevplt = axrvar_.plot(np.arange(1.5, len(Ddev)+1, 1.0),
                               Ddev, color='blue', lw=1.25,
                               label='DV (' + str(height) + ' most abundant)')
        axrvar_.get_yaxis().labelpad = 15
        axrvar_.set_ylabel('Differences Variability',
                           color='blue', rotation=270)
        plt.setp(axrvar_.yaxis.get_ticklabels(), size='small')
        # Make the y-axis label and tick labels match the line color.
        for tl in axrvar_.get_yticklabels():
            tl.set_color('blue')
        axrvar_.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axrvar_.yaxis.set_major_locator(MultipleLocator(0.1))
        axrvar_.yaxis.set_minor_locator(MultipleLocator(0.02))
        axrvar_.set_ylim(ymin=0, ymax=0.8)
        axrvar_.set_xlim(xmin=0, xmax=width)
        # Legend: Add the plots in desired order
        pltsum = smaxplt + Ddevplt
        labels = [l.get_label() for l in pltsum]
        axrvar_.legend(pltsum, labels, fontsize='x-small',
                       bbox_to_anchor=(0, 0), borderaxespad=1.0)

        # Rank colorbar
        cbar = self.fig.colorbar(pcm, ax=axrank, fraction=0.1, pad=0.01,
                                 shrink=0.98, aspect=20, extend='both')
        cbardummy = self.fig.colorbar(pcm, ax=axrvar, fraction=0.1, pad=0.01,
                                      shrink=0.98, aspect=20, extend='both')
        self.fig.delaxes(cbardummy.ax)
        cbardummy = self.fig.colorbar(pcm, ax=axrvar_, fraction=0.1, pad=0.01,
                                      shrink=0.98, aspect=20, extend='both')
        self.fig.delaxes(cbardummy.ax)
        cbar.ax.invert_yaxis()
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.get_yaxis().labelpad = -8
        cbar.ax.set_ylabel('Accumulated abundance rank', size='medium',
                           weight='semibold', color='white', rotation=270)
        cbar.ax.annotate(
            '1', xy=(0.5, 0),
            horizontalalignment='center', verticalalignment='center',
            size='small', weight='bold', color='black'
        )
        cbar.ax.annotate(
            str(height), xy=(0.5, 1),
            horizontalalignment='center', verticalalignment='center',
            size='small', weight='bold', color='white'
        )

        # Rank Stability Index (see wiki doc for details) subplot
        plt.setp(axRSI.get_xticklines(), visible=False)
        plt.setp(axRSI.get_yticklines(), visible=False)
        Cheight, Cwidth = C.shape
        maxHop = (Cwidth - 1) * (Cheight - 1)
        RSIpwr = 4.0
        R['RSI'] = np.power(1 - D.abs().sum(axis='columns') / maxHop, RSIpwr)
        RSI = R['RSI'].values
        axRSI.pcolormesh(np.transpose([RSI]), cmap=plt.cm.inferno,
                         shading='flat',  # vmin=0.0, vmax=1.0,
                         norm=colors.PowerNorm(gamma=RSIpwr)
                         )
        # Annotate the stability column with the RSI

        RSIcolorlimit = min(RSI) + ((RSIpwr - 1) / RSIpwr) * (
            max(RSI) - min(RSI))
        # Select color for the RSI value in the plot
        for y in range(height):
            if R['RSI'][y] > RSIcolorlimit:
                color = 'black'
            else:
                color = 'white'
            axRSI.annotate(
                r'100' if R['RSI'][y] > 0.999 else r'%2.1f' % (R['RSI'][y] *
                                                               100),
                xy=(0.5, y + 0.5),
                horizontalalignment='center', verticalalignment='center',
                size='x-small', color=color
            )
        axRSI.set_xticks([])
        axRSI.get_yaxis().labelpad = 15
        axRSI.set_ylabel('Rank Stability Index (in %)', rotation=270)
        axRSI.yaxis.set_label_position("right")

        # Store the rank+RSI dataframe for potential output
        self.rankdf = R.iloc[::-1]  # 1st ranked will be in 1st row
        # Final plotting step
        super(RankPlot, self).end_plot()


class AbsFreqPlot(CCplot):
    """Do a 2D histogram of the difference of data with its mean."""

    def __init__(
            self, ydata,
            xlabel='Number of time sample',
            ylabel='Absolute frequency of elements',
            path='.', name='', format='png'
    ):
        # Call superclass initialization
        super(AbsFreqPlot, self).__init__(
            xlabel, ylabel, xscale='nogrid', yscale='nogrid',
            path=path, name=name, plot='_AbsFreqPlot', format=format
        )
        # Do the specific staff
        leny = len(ydata)
        n = np.arange(leny)
        ymax = ydata.max()
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0.0, vmax=ymax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        colorValues = scalarMap.to_rgba(ydata)
        self.ax.set_xlim(xmin=0, xmax=leny)
        self.ax.set_ylim(ymin=0, ymax=ymax)
        self.title = self.name + ': Absolute Frequencies Plot'
        # Plot the graphs
        plt.bar(n, ydata, width=1.0, color=colorValues)
        if leny <= 20:
            plt.xticks(n + 0.5, n + 1)
        super(AbsFreqPlot, self).end_plot()


class ZRFplot(CCplot):
    """
    Bar plot of relative frequency of zero among the elements.

    For example, the bar over 0.0 counts the number of elements that
    have all their data over zero. The bar over 1.0 would count the
    number of elements that have all their data null, but such a count
    is always zero as those elements are filtered in the first stages
    of the data processing in cmplxcruncher.
    """

    def __init__(
            self, series,
            xlabel='ZRF (Zero Relative Frequency)',
            ylabel='Number of elements',
            path='.', name='', format='png'
    ):
        # Call superclass initialization
        super(ZRFplot, self).__init__(
            xlabel, ylabel, xscale='nogrid', yscale='nogrid',
            path=path, name=name, plot='_ZRFhist', format=format
        )
        # Do the specific staff (normed hist not available: mpl reported BUG!)
        N, bins, patches = plt.hist(series, bins=20, range=(0, 1))
        # range of the colormap
        fracs = N.astype(float) / N.max()
        cNorm = colors.Normalize(vmin=fracs.min(), vmax=fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = cmx.cool(cNorm(thisfrac))
            thispatch.set_facecolor(color)
        ymin, ymax = plt.ylim()
        # plt.vlines(0.2, ymin=0, ymax=ymax, colors='g', linestyles='dotted')
        plt.fill_between([0, 0.2], ymax, color='g', alpha=0.1)
        plt.text(
            0.15, 0.9 * ymax, 'core region',
            rotation=90, fontsize=14, color='g'
        )
        self.ax.set_xlim(xmin=0, xmax=1)
        self.ax.set_ylim(ymin=0, ymax=ymax)
        self.ax.set_xmargin(0)
        self.ax.set_ymargin(0)
        self.title = self.name + ': Relative frequency of zero'
        # Plot the graphs
        super(ZRFplot, self).end_plot()


class FitPowPlot(CCplot):
    """
    Power-law fitting.

    It may be best done by 1st converting to a linear equation
    and then fitting to a straight line:
      y = K * x^b
      log(y) = log(K) + b*log(x)
    """

    def __init__(
            self, x=None, y=None, xerr=None,
            xlabel='mean',
            ylabel='std',
            path='.', name='',
            format='png', verbose=False, test=False
    ):
        # Input data for calcs
        self.pfm = PowerFitModel(x=x, y=y, xerr=xerr, verbose=verbose)
        # Input data for plotting
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.path = path
        self.name = name
        self.format = format
        self.verbose = verbose
        # Test data
        self.ampTest = 0.0
        self.indexTest = 0.0
        self.doTest = False
        self.doneTest = False
        self.pfmBkp = None
        if test:
            self.setupTest(test)

    def setupTest(
            self, errormodel, amp=0.5, index=0.5,
            scale=5, points=50, loadlast=False, bxerr=False
    ):
        """Setup test mode."""
        # Back-up the reference to the power fit model and create
        #  another void for the test
        if not self.doneTest:
            self.pfmBkp = self.pfm
            self.pfm = PowerFitModel(verbose=self.verbose)
        if loadlast:
            self.ampTest = self.pfmBkp.amp
            self.indexTest = self.pfmBkp.index
        else:
            self.ampTest = amp
            self.indexTest = index
        self.pfm.setupTest(
            errormodel, amp=amp, index=index,
            scale=scale, points=points, bxerr=bxerr
        )
        # Set the test flags
        self.doTest = True
        self.doneTest = False

    def disableTest(self):
        """Disable test mode."""
        # Recover the data reference and delete test data
        del (self.pfm)
        self.pfm = self.pfmBkp
        self.pfmBkp = None
        self.ampTest = 0.0
        self.indexTest = 0.0
        # Set the test flags
        self.doTest = False
        self.doneTest = False

    def fit(self, model='', bootall=False, fulloutput=False, p0=None):
        """Fit the data using the selected model."""
        self.pfm.clear_fit()
        if model == 'IWLLR':
            self.pfm.fitInvW(loglog=True)
        elif model == 'IWNLR':
            self.pfm.fitInvW(loglog=False)
        elif model == 'xWboot':
            pfm_best = PowerFitModel.bestFit2(
                self.pfm.xdata, self.pfm.ydata, xerr=self.pfm.xerr,
                name=self.pfm.name, verbose=self.verbose
            )
            del (self.pfm)
            self.pfm = pfm_best
        else:
            if bootall:
                print('\n### ORDINARY BOOTSTRAP ###')
                self.pfm.clear_fit()
                self.pfm.fit(model)
                self.pfm.bootstrap()
                print('\n### MC BOOTSTRAP ###')
                self.pfm.clear_fit()
                self.pfm.fit(model)
                self.pfm.bootstrapMC()
                print('\n### RESIDUALS RESAMPLING BOOTSTRAP ###')
                self.pfm.clear_fit()
                self.pfm.fit(model)
                self.pfm.bootstrapRR()
                print('\n### WILD(NORMAL) BOOTSTRAP ###')
                self.pfm.clear_fit()
                self.pfm.fit(model)
                self.pfm.bootstrapWild(dist='normal')
                print('\n### WILD(RADEMACHER) BOOTSTRAP ###')
                self.pfm.clear_fit()
                self.pfm.fit(model)
                self.pfm.bootstrapWild(dist='rademacher')
            else:
                if p0 is None:
                    self.pfm.fit(model)
                else:
                    self.pfm.fit(model, p0=p0)
        # if not bootall:
        # self.pfm.bootstrap() # Default bootstrap method
        # Was it a test? Then, done!
        if self.doTest:
            self.doneTest = True
        if fulloutput:
            return (self.pfm.get(full=True))
        else:
            return (self.pfm.get())

    def bestFit(
            self,
            plot=True, resid=True,
            fulloutput=False, legacy=False
    ):
        """Wrapper for PowerFitModel.bestFit methods."""
        pfm_best = PowerFitModel.bestFit2(
            self.pfm.xdata, self.pfm.ydata,
            xerr=self.pfm.xerr, forceboot=False,
            name=self.pfm.name, verbose=self.verbose,
            legacy=legacy
        )
        del (self.pfm)
        self.pfm = pfm_best
        # Was it a test? Then, done!
        if self.doTest:
            self.doneTest = True
        if plot:
            self.plotLin()
            self.plotLog()
            if resid:
                self.plotResid()
        if fulloutput:
            return (self.pfm.get(full=True))
        else:
            return (self.pfm.get())

    def prePlot(self):
        """Prepare the plot and the title label before plotting."""
        self.plot = '_' + self.ylabel + 'VS' + self.xlabel
        if self.doneTest:
            self.name = self.pfm.name
        if self.pfm.isModel('LLR'):
            if self.doneTest:
                self.title = (self.name + ': Log-LR p-law fit V=' +
                              ('%5.2f' % self.ampTest) + r' $\beta$=' +
                              ('%5.2f' % self.indexTest))
            else:
                self.title = self.name + ': Log-LR pwr-law fit'
        elif self.pfm.isModel('NLR'):
            if self.doneTest:
                self.title = (self.name + ': Non-LR p-law fit V=' +
                              ('%5.2f' % self.ampTest) + r' $\beta$=' +
                              ('%5.2f' % self.indexTest))
            else:
                self.title = self.name + ': Non-LR pwr-law fit'
        elif self.pfm.isModel('AVG'):
            if self.doneTest:
                self.title = (self.name + ': AVR-R p-law fit V=' +
                              ('%5.2f' % self.ampTest) + r' $\beta$=' +
                              ('%5.2f' % self.indexTest))
            else:
                self.title = self.name + ': Averaged NonLR/LogLR pwr-law fit'
        elif self.pfm.isModel('IWLLR'):
            self.title = self.name + ': Inverted-Weighted-LogLR p-law fit'
        elif self.pfm.isModel('IWNLR'):
            self.title = self.name + ': Inverted-Weighted-NonLR p-law fit'
        elif self.pfm.isModel('xWLLR'):
            self.title = self.name + ': x-Weighted-LogLR pwr-law fit'
        elif self.pfm.isModel('xWNLR'):
            self.title = self.name + ': x-Weighted-NonLR pwr-law fit'
        elif self.pfm.isModel('xWAVG'):
            self.title = self.name + ': Averaged x-Wght-NLR/LLR p-law fit'
        elif self.pfm.isModel('xWboot'):
            self.title = self.name + ': bootstrap x-Weighted pwr-law fit'
        else:
            self.title = (self.name +
                          ': In FitPowPlot.prePlot(), model not implemented!!')
        self.plot += ('_' + self.pfm.getModel())

    def plotLin(self):
        # Call prePlot()
        self.prePlot()
        # Call superclass initialization
        super(FitPowPlot, self).__init__(
            self.xlabel, self.ylabel, xscale='lin', yscale='lin',
            path=self.path, name=self.name, plot=self.plot, format=self.format,
            title=self.title
        )

        # Do the lineal plots
        self.plot += '_LIN'
        # plt.axis('scaled') # ADDED FOR LIN PLOTS!!!
        xmin = np.min(self.pfm.xdata)
        xmax = np.max(self.pfm.xdata)
        ymin = np.min(self.pfm.ydata)
        ymax = np.max(self.pfm.ydata)
        x = np.linspace(xmin, xmax, num=100)
        if self.doneTest:
            plt.plot(x, self.pfm.powerlaw(x, self.ampTest, self.indexTest),
                     c='g')  # Test power-law
            plt.scatter(
                self.pfm.xdata, self.pfm.ydata,
                s=15, c='0.30'
            )  # Test Data
        else:
            plt.scatter(self.pfm.xdata, self.pfm.ydata, s=15,
                        c='r')  # Input Data
        plt.plot(x, self.pfm.powerlaw(x, self.pfm.amp, self.pfm.index),
                 c='b')  # Fit
        if (self.pfm.xerr is not None and not self.pfm.isModel('NLR') and
                not self.pfm.isModel('LLR') and not self.pfm.isModel('AVG')):
            plt.errorbar(self.pfm.xdata, self.pfm.ydata, xerr=self.pfm.xerr,
                         fmt='k.')  # Data

        # Legend (fit and R^2)
        plt.text(
            xmin, ymax, r'$Vx^\beta=(%5.2f\pm %5.2f)x^{%5.2f\pm %5.2f}$' % (
                self.pfm.amp, abs(self.pfm.ampErr),
                self.pfm.index, abs(self.pfm.indexErr)
            ),
            fontsize=20
        )
        plt.text(
            xmin, ymax - 0.2 * (ymax - ymin),
                  r'$\bar R^2=%1.6f$' % (self.pfm.R2),
            fontsize=20
        )
        # Legend (weights in case of averaged model)
        if self.pfm.isModel('AVG') or self.pfm.isModel('xWAVG'):
            posx = xmin + 0.50 * (xmax - xmin)
            posy = ymin + 0.01 * (ymax - ymin)
            plt.text(
                posx, posy,
                r'$w_{\mathrm{norm}}=%1.2f\quad w_{\mathrm{logn}}=%1.2f$' % (
                    self.pfm.w_norm, self.pfm.w_logn
                ),
                fontsize=14
            )

        # Call superclass finalization
        super(FitPowPlot, self).end_plot()

        # Recover plot name
        self.plot = self.plot[0:-4]

    def plotLog(self):
        # Call prePlot():
        self.prePlot()
        # Call superclass initialization
        super(FitPowPlot, self).__init__(
            self.xlabel, self.ylabel, xscale='log', yscale='log',
            path=self.path, name=self.name, plot=self.plot, format=self.format,
            title=self.title
        )

        # Do the loglog plots
        self.plot += '_LOG'
        xmin = np.min(self.pfm.xdata)
        xmax = np.max(self.pfm.xdata)
        ymin = np.min(self.pfm.ydata)
        ymax = np.max(self.pfm.ydata)
        lxmin = np.log10(xmin)
        lxmax = np.log10(xmax)
        x = np.logspace(lxmin, lxmax, num=100)
        if self.doneTest:
            plt.loglog(x, self.pfm.powerlaw(x, self.ampTest, self.indexTest),
                       c='g')  # Test power-law
            plt.scatter(self.pfm.xdata, self.pfm.ydata, s=15,
                        c='0.30')  # Test Data
        else:
            plt.scatter(self.pfm.xdata, self.pfm.ydata, s=15,
                        c='r')  # Input Data
        plt.loglog(x, self.pfm.powerlaw(x, self.pfm.amp, self.pfm.index),
                   c='b')  # Fit
        if (self.pfm.xerr is not None and
                not self.pfm.isModel('NLR') and
                not self.pfm.isModel('LLR') and
                not self.pfm.isModel('AVG')):
            plt.errorbar(self.pfm.xdata, self.pfm.ydata, xerr=self.pfm.xerr,
                         fmt='k.')  # Data

        # Legend (fit and R^2)
        self.ax.set_xlim(left=xmin / 5, right=x[-1] * 5)
        self.ax.set_ylim(bottom=ymin / 10, top=ymax * 10)
        plt.text(
            xmin, ymax / 2,
                  r'$Vx^\beta=(%5.2f\pm %5.2f)x^{%5.2f\pm %5.2f}$' % (
                      self.pfm.amp, abs(self.pfm.ampErr),
                      self.pfm.index, abs(self.pfm.indexErr)
                  ),
            fontsize=20
        )
        plt.text(xmin, ymax / 5,
                 r'$\bar R^2=%1.6f$' % self.pfm.R2, fontsize=20)
        # Legend (weights in case of averaged model)
        if self.pfm.isModel('AVG') or self.pfm.isModel('xWAVG'):
            posx = 10 ** (lxmin + 0.5 * (lxmax - lxmin))
            posy = ymin
            plt.text(
                posx, posy,
                r'$w_{\mathrm{norm}}=%1.2f\quad w_{\mathrm{logn}}=%1.2f$' % (
                    self.pfm.w_norm, self.pfm.w_logn),
                fontsize=14)

        # Call superclass finalization
        super(FitPowPlot, self).end_plot()

        # Recover plot name
        self.plot = self.plot[0:-4]

    def plotResid(self, plot=True):
        """Plot diagnostics for the residues of a fit."""

        # Call prePlot():
        self.prePlot()

        # If no residues or errors, plot only ErrorPlot with a message
        if (self.pfm.RMSE < _EPS) or ((self.pfm.errors ** 2).sum() < _EPS):
            ErrorPlot(xlabel='', ylabel='',
                      xscale='none', yscale='none',
                      path=self.path, name=self.name,
                      plot=(self.plot + '_RES'), format=self.format,
                      title=(self.name + ': residues analysis of ' +
                             self.pfm.getModel() + '-model fit'),
                      msg="The fit seems perfect!\n The residues vanished.")
            return

        # Mean and unbiased std of residues
        mu = np.mean(self.pfm.resids)
        sigma = np.std(self.pfm.resids, ddof=1)

        # normality=stools.omni_normtest(self.pfm.resids)
        # heteroskedasticity=stools.jarque_bera(self.resid)
        # print(stats.shapiro(rself.resid))

        if self.verbose:
            print('\n<> ccPlot.FitPow.plotResid <> Model used for the fit : ' +
                  self.pfm.getModel())
            # print('<> OmniBusNorm-Test =',normality)
            # print('<> JarqueBera-Test =',stools.jarque_bera(self.pfm.resids))
            print('<> Shapiro-Test = ', stats.shapiro(self.pfm.resids))

        if (plot):
            # Don't call superclass initialization, instead do custom init.
            self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 9))
            self.fig.subplots_adjust(
                left=0.10, bottom=0.05, right=0.95,
                top=0.90, wspace=0.3, hspace=0.3
            )
            self.fig.suptitle(self.name + ': residues analysis of ' +
                              self.pfm.getModel() + '-model fit', fontsize=18)
            self.plot += '_RES'

            # [0,0] Simple residues plot
            self.ax[0, 0].set_xscale('log', nonposx='clip')
            xleft = 10 ** np.floor(np.log10(min(self.pfm.xdata)))
            xright = 10 ** np.ceil(np.log10(max(self.pfm.xdata)))
            self.ax[0, 0].set_xlim(left=xleft, right=xright)
            self.ax[0, 0].grid()
            self.ax[0, 0].stem(
                self.pfm.xdata, self.pfm.resids,
                linefmt='r-', markerfmt='ro', basefmt='g-'
            )
            self.ax[0, 0].set_xlabel(self.xlabel + ' (log scale)')
            self.ax[0, 0].set_ylabel('Residues (model scale)', fontsize=12)
            self.ax[0, 0].set_title('Model residues plot')

            # [0,1] Normal quantiles plot
            # self.fig.sca(self.ax[0,1])
            (osm, osr), (slope, intercept, r) = stats.probplot(
                self.pfm.resids, sparams=(mu, sigma), dist="norm", plot=None
            )
            self.ax[0, 1].axis('equal')
            self.ax[0, 1].grid()
            self.ax[0, 1].set_xlabel('Quantiles (normal distribution)')
            self.ax[0, 1].set_ylabel('Residues (model scale)')
            self.ax[0, 1].set_title('Probability plot of (ordered) residues')
            self.ax[0, 1].plot(osm, osr, 'ro', osm, slope * osm + intercept)
            xmin = min(osm)
            xmax = max(osm)
            ymin = osr[0]
            ymax = osr[-1]
            posx = xmin + 0.70 * (xmax - xmin)
            posy = ymin + 0.01 * (ymax - ymin)
            self.ax[0, 1].text(posx, posy, r'$R^2=%1.4f$' % r)
            self.ax[0, 1].apply_aspect()
            [xmin01, xmax01, ymin01, ymax01] = self.ax[0, 1].axis()

            # [1,0] Accumulated residue plot
            self.ax[1, 0].set_xscale('log', nonposx='clip')
            self.ax[1, 0].set_xlim(left=xleft, right=xright)
            xargsort = self.pfm.xdata.argsort()
            cumESS = np.cumsum(np.append(self.pfm.errors[xargsort], 0.0) ** 2)
            self.ax[1, 0].fill_between(
                np.append(self.pfm.xdata[xargsort], 1.0),
                cumESS, color='r', alpha=0.3
            )
            self.ax[1, 0].grid()
            self.ax[1, 0].set_xlabel(self.xlabel + ' (log scale)')
            self.ax[1, 0].set_ylabel('Accumulated ESS (lin scale)')
            self.ax[1, 0].set_title('Accumulated ESS plot')

            # [1,1] Residues normal histogram plot
            num_bins = 15
            # the histogram of the data
            n, bins, patches = self.ax[1, 1].hist(
                self.pfm.resids, num_bins,
                normed=True, facecolor='green', alpha=0.5
            )
            #  add a 'best fit' line
            x = np.linspace(xmin01, xmax01, num=num_bins * 5)
            y = mlab.normpdf(x, mu, sigma)
            self.ax[1, 1].plot(x, y, 'r--')
            self.ax[1, 1].set_xlim(left=xmin01, right=xmax01)
            self.ax[1, 1].grid()
            self.ax[1, 1].set_title('Normality plot of the residues')
            self.ax[1, 1].set_xlabel('Residues')
            self.ax[1, 1].set_ylabel('Probability density')

            # Call superclass finalization
            super(FitPowPlot, self).end_plot()

            # Recover plot name
            self.plot = self.plot[0:-4]


class SummaryPlot(CCplot):
    """Plot a monochrome scattered error plot."""

    def __init__(self, xdata, xerr, ydata, yerr, xlabel='V', ylabel='beta',
                 path='.', name='', format='png'):
        # Call superclass initialization
        super(SummaryPlot, self).__init__(
            xlabel, ylabel, xscale='lin', yscale='lin',
            path=path, name=name, plot='_SUMMARYplot', format=format
        )
        # Do ALL the staff
        plt.errorbar(xdata, ydata, yerr, xerr, fmt='.b')
        self.title = self.name + ': Fit Summary Plot [monochrome]'
        super(SummaryPlot, self).end_plot()


class ColorSummaryPlot(CCplot):
    """Plot a colored scattered error plot."""

    def __init__(self,
                 xlabel=r'$V$', ylabel=r'$\beta$',
                 path='.', name='', format='png'):
        # Call superclass initialization
        super(ColorSummaryPlot, self).__init__(
            xlabel, ylabel, xscale='nogrid', yscale='nogrid',
            path=path, name=name, plot='_Summary', format=format
        )
        # Initialize data members
        self._pointscounter = 0
        self._color = itertools.cycle((
            'c', 'm', 'y', 'k', 'r', 'g', 'b', '0.30'))
        self._marker = itertools.cycle(('o', 'v', '^', '8', 's', '*', 'D'))
        # self.__linestyle = itertools.cycle(('-','--',':'))

    def add_point(self, xdata, xerr, ydata, yerr, label):
        """Add a series of points with error bars to the plot."""
        self.ax.errorbar(
            xdata, ydata,
            yerr=yerr.values, xerr=xerr.values,
            color=next(self._color), marker=next(self._marker),
            label=label, fmt='.'
        )
        self._pointscounter += 1

    def end_plot(self):
        """Plot to screen or save the figure."""
        if self.name == '':
            self.name = 'cplxCrnch'
            self.title = 'Overall cmplxcruncher Fit Summary Plot'
        elif ('Weighted' in self.name):
            self.title = ('Overall ' + self.name +
                          ' cmplxcruncher Fit Summary')
            self.name = 'cplxCrnch_' + self.name
            if ('_STAN' in self.name):
                self.ax.add_artist(Ellipse((0, 0), 4, 4, color='b', alpha=0.08,
                                           linestyle='dashed'))
                self.ax.add_artist(Ellipse((0, 0), 2, 2, color='m', alpha=0.3))
                self.ax.axis('equal')
        else:
            self.title = self.name + ': Fit Summary Plot'
        try:  # Avoid mpl error due to empty legend
            self.ax.legend(loc='best', prop={'size': 'x-small'})
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass
        super(ColorSummaryPlot, self).end_plot()


def ChkDir(path):
    """Make dir and check for existence and w&x permissions."""
    error = False
    try:
        os.makedirs(path)
    except (OSError):
        if not os.access(path, (os.W_OK | os.X_OK)):
            print('\nWARNING! Unable to write results in directory "' +
                  path + '"...')
            error = True
    return (error)


def DoAllPlots(A, S, Asum, C, name, flgs):
    """Call the plot functions."""
    # Extract local variables from flgs
    format = flgs['format']
    path = flgs['respath']
    lessplot = flgs['lessplot']
    moreplot = flgs['moreplot']
    DFcols = flgs['DFcols']
    if flgs['debug']:
        enableDevMode()
    else:
        disableDevMode()
    # Initialize output variables
    DFrow = []
    correlms = None
    corrtime = None
    rank = None
    # Define and check paths for the plots
    path1 = []
    if not path == '' and not format == '':
        path1 = path + PATH_FITS
        if (ChkDir(path1)):
            path1 = path
        pathname1 = path1 + name
    else:
        pathname1 = name
    try:
        fpp = FitPowPlot(
            S['1-mean'], S['2-std'], name=name,
            path=path1, format=format, verbose=devflag
        )
        (amp, ampErr, index, indexErr, R2, pcorr, model) = fpp.bestFit()
        pass
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print('\t\tERROR! Failed the NOT-WEIGHTED plots of std VS mean in ' +
              pathname1)
        if devflag:
            raise
        else:
            ErrorPlot(xlabel='mean', ylabel='std',
                      xscale='log', yscale='log',
                      path=path1, name=name,
                      plot='_stdVSmeanNoErr', format=format,
                      title=(name +
                             ': data fit to power law (using scicy.leastsq)')
                      )
    try:
        fpp = FitPowPlot(
            S['1-mean'], S['2-std'], xerr=S['5-sem'], name=name,
            path=path1, format=format, verbose=devflag
        )
        (xW_amp, xW_ampErr, xW_index,
         xW_indexErr, xW_R2, xW_pcorr, xW_model) = fpp.bestFit(resid=False)

        # If debug mode is enabled, fit further methods and models
        if devflag and not lessplot:
            print("\n >>> ALSO TRY LEGACY x-Weighted BEHAVIOUR <<< ")
            try:
                fpp.bestFit(legacy=True)
            except CovMatrixError:
                pass
            print("\n >>> ALSO TRY IWLLR MODEL <<< ")
            fpp.fit(model='IWLLR')
            fpp.plotLin()
            fpp.plotLog()
            fpp.plotResid()

        # Sum calculations
        absfmean = Asum.mean()
        absfstd = Asum.std()
        absfsum = Asum.sum()

        # Function output arrangement
        DFrow = pd.DataFrame.from_items([(name, [
            (A.index).size, (A.columns).size, absfmean, absfstd, absfsum,
            amp, ampErr, index, indexErr, R2, pcorr, model,
            xW_amp, xW_ampErr, xW_index, xW_indexErr, xW_R2, xW_pcorr,
            xW_model, None, None, None, None
        ])],
                                        orient='index',
                                        columns=DFcols
                                        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print('\t\t ERROR! Failed the x-weighted plots of std VS mean in ' +
              pathname1)
        if devflag:
            raise
        else:
            ErrorPlot(xlabel='mean', ylabel='std',
                      xscale='log', yscale='log',
                      path=path1, name=name,
                      plot='_stdVSmean', format=format,
                      title=name + ': data fit to power law'
                      )

    if not lessplot:
        path2 = pathname2 = path4 = pathname4 = []
        if not path == '' and not format == '':
            path2 = path + PATH_HIST
            if (ChkDir(path2)):
                path2 = path
            pathname2 = path2 + name
            path4 = path + PATH_CORK
            if (ChkDir(path4)):
                path4 = path
            pathname4 = path4 + name
        else:
            pathname2 = name
            pathname4 = name
        try:
            Hist2Dplot(A, S, path=path2, name=name, format=format)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\t\t ERROR! Failed the 2D histogram plot in ' + pathname2)
            if devflag:
                raise
            ErrorPlot(xlabel=r'$\log_{10}\left(\bar{x}\right)$',
                      ylabel=r'$\Delta\bar x$',
                      xscale='lin', yscale='lin',
                      path=path2, name=name,
                      plot='_hist2D', format=format,
                      title=name + ': 2D deviation plot')
        try:
            AbsFreqPlot(Asum.values, path=path2, name=name, format=format)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\t\t ERROR! Failed the plot of absolute frequency in ' +
                  pathname2)
            if devflag:
                raise
            ErrorPlot(xlabel='Number of time sample',
                      ylabel='Absolute frequency of elements',
                      xscale='lin', yscale='lin',
                      path=path2, name=name,
                      plot='_AbsFreqPlot', format=format,
                      title=name + ': Absolute Frequencies Plot')
        try:
            ZRFplot(S['6-ZRF'], path=path2, name=name, format=format)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\t\t ERROR! Failed histogram of Zero Relative Freq in ' +
                  pathname2)
            if devflag:
                raise
            ErrorPlot(xlabel='Normalized frequency of zero',
                      ylabel='Number of elements',
                      xscale='lin', yscale='lin',
                      path=path2, name=name,
                      plot='_NFZhist', format=format,
                      title=name + ': Normalized Zero Frequency')
        try:
            correlms = CorrPlot(C, 'elements', maxweight=1.1,
                                path=path4, name=name, format=format).corrdf
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\t\t ERROR! Failed the elements corr matrix plot in ' +
                  pathname4)
            if devflag:
                raise
            ErrorPlot(xlabel='', ylabel='',
                      xscale='none', yscale='none',
                      path=path4, name=name,
                      plot='_ElementsCorr', format=format,
                      title=name + ': elements corr matrix')
        try:
            rank = RankPlot(C, path=path4, name=name, format=format).rankdf
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('\t\t ERROR! Failed the rank matrix and stability plot in ' +
                  pathname4)
            if devflag:
                raise
            ErrorPlot(xlabel='', ylabel='',
                      xscale='none', yscale='none',
                      path=path4, name=name,
                      plot='_Rank', format=format,
                      title=name + ': rank matrix & stability')
        if moreplot:
            path3 = pathname3 = []
            if not path == '' and not format == '':
                path3 = path + PATH_OTHR
                if (ChkDir(path3)):
                    path3 = path
                pathname3 = path3 + name
            else:
                pathname3 = name
            try:
                SemilogxPlot(
                    S['1-mean'], S['3-skew'],
                    xlabel='mean', ylabel='skew',
                    path=path3, name=name, format=format
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('\t\t ERROR! Failed the plot of skew VS mean in ' +
                      pathname3)
                if devflag:
                    pass  # It was raise
                ErrorPlot(xlabel='mean', ylabel='skew',
                          xscale='log', yscale='lin',
                          path=path3, name=name,
                          plot='_skewVSmean', format=format,
                          title=name + ': skew plot')
            try:
                SemilogxPlot(
                    S['1-mean'], S['4-kurt'],
                    xlabel='mean', ylabel='kurt',
                    path=path3, name=name, format=format
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('\t\t ERROR! Failed the plot of kurtosis VS mean in ' +
                      pathname3)
                if devflag:
                    pass  #  It was raise
                ErrorPlot(xlabel='mean', ylabel='kurt',
                          xscale='log', yscale='lin',
                          path=path3, name=name,
                          plot='_kurtVSmean', format=format,
                          title=name + ': kurt plot')
            try:
                corrtime = CorrPlot(C, 'times', maxweight=1.1,
                                    path=path4, name=name,
                                    format=format).corrdf
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('\t\t ERROR! Failed the times correlation plot in ' +
                      pathname4)
                if devflag:
                    raise
                ErrorPlot(xlabel='', ylabel='',
                          xscale='none', yscale='none',
                          path=path4, name=name,
                          plot='_TimesCorr', format=format,
                          title=name + ': times corr matrix')
    return (DFrow, correlms, corrtime, rank)


def DoTestPlots(path, format):
    """Call the test plot functions (under development!)."""
    path1 = pathname1 = []
    name = 'ccTEST'
    if not path == '' and not format == '':
        path1 = path + PATH_FITS
        if (ChkDir(path1)):
            path1 = path
        pathname1 = path1 + name
    else:
        pathname1 = name
    # Fits and plots
    fpp = FitPowPlot(path=path1, format=format, verbose=devflag)
    errormodels = ["mixed", "add", "mul", "non"]  # Also available: "bad"
    for errormodel in errormodels:
        print("\n !!!! TEST " + errormodel.upper() + " NOISE !!!! ")
        for bxerr in [False, True]:
            print("\n \\/>/>/> " + ('W' if bxerr else 'Un-w') +
                  "eighted </</</\\")
            fpp.setupTest(errormodel=errormodel, bxerr=bxerr)
            fpp.bestFit(legacy=True)
            if bxerr is True:
                print("\n >>> ALSO TEST HERE xWboot MODEL <<< ")
                fpp.fit(model='xWboot')
                fpp.plotLin()
                fpp.plotLog()
                fpp.plotResid()
            if errormodel == "mul" and bxerr is True:
                print("\n >>> ALSO TEST HERE IWLLR MODEL WITH MULT NOISE <<< ")
                fpp.fit(model='IWLLR')
                fpp.plotLin()
                fpp.plotLog()
                fpp.plotResid()

    # raise(Exception("Development-TEST release exception: FINISH HERE!!!"))

    path2 = pathname2 = path4 = pathname4 = []
    if not path == '' and not format == '':
        path2 = path + PATH_HIST
        if (ChkDir(path2)):
            path2 = path
        pathname2 = path2 + name
        path4 = path + PATH_CORK
        if (ChkDir(path4)):
            path4 = path
        pathname4 = path4 + name
    else:
        pathname2 = name
        pathname4 = name
    # TO DO: Insert plots here
    path3 = pathname3 = []
    if not path == '' and not format == '':
        path3 = path + PATH_OTHR
        if (ChkDir(path3)):
            path3 = path
        pathname3 = path3 + name
    else:
        pathname3 = name
        # TO DO: Insert plots here
