"""
This module provides the models and data classes of the package.

"""

# Packages
import sys
import numpy as np
from scipy import optimize, stats
from cmplxcruncher.config import *
# import statsmodels.stats.stattools as stools
# import statsmodels.stats.diagnostic as sdiags
# Uncommenting next 2 lines needed for scatter plot in PowerFitModel.bestFit2
# from pandas import DataFrame
# from pandas.tools.plotting import scatter_matrix

# Constants
_EPS = np.finfo(np.double).eps


class ModelError(Exception):

    """Base class for exceptions in the :mod:`cmplxcruncher.models` module."""

    def __init__(self, message):
        super().__init__(message)


class UnknownModelError(ModelError):

    """Raised when a model label is unknown."""

    def __init__(self, instance, frame, model):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Unknown model ' +
            repr(model)
        )


class UnsupportedModelError(ModelError):

    """Raised when a model is still unsupported in a method."""

    def __init__(self, instance, frame, model):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Model ' + repr(model) + ' is not supported by this method.'
        )


class UnfittedModelError(ModelError):

    """Raised when trying to use a model that is not fitted yet."""

    def __init__(self, instance, frame):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Model not fitted yet!'
        )


class UnknownResidsError(ModelError):

    """Raised when the kind of residues is unknown."""

    def __init__(self, instance, frame, resids):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Unknown kind of residues ' +
            repr(resids)
        )


class MissingParamsError(ModelError):

    """Raised when trying to use a fitted model but with missing parameters."""

    def __init__(self, instance, frame):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Fitted model but missing parameters!'
        )


class UnknownErrorModelError(ModelError):

    """Raised when the error-model is unknown."""

    def __init__(self, instance, frame, errormodel):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': Unknown error-model ' + repr(errormodel) +
            '\n\t(options: \'non(e)\', \'add(itive)\', ' +
            '\'mul(tiplicative)\', \'mix(ed)\')'
        )


class CovMatrixError(ModelError):

    """Raised when the cov matrix of a fit is scalar."""

    def __init__(self, instance, frame, model, cov):
        super().__init__(
            '\nERROR in ' +
            repr(instance.__module__ + '.' +
                 instance.__class__.__name__ + '.' +
                 frame.f_code.co_name + '()') +
            ': While fitting the model' + repr(model) +
            ', covariance matrix resulted scalar ' + repr(cov)
        )


class PowerFitModel(object):

    """
    Model for a power-law fit.
    """

    def __init__(
            self, x=None, y=None, xerr=None,
            name='PowerFitModel', verbose=False
    ):
        # List of models supported (to all or some extent)
        self.supported = [
            'LLR', 'NLR', 'AVG',
            'IWLLR', 'IWNLR',
            'xWLLR', 'xWNLR', 'xWAVG',
            'xWboot'
        ]
        try:
            self.xdata = np.core.numeric.asarray(x.values) + 0.0
        except:
            if x is not None:
                self.xdata = np.core.numeric.asarray(x)
            else:
                self.xdata = None
        try:
            self.ydata = np.core.numeric.asarray(y.values) + 0.0
        except:
            if y is not None:
                self.ydata = np.core.numeric.asarray(y)
            else:
                self.ydata = None
        try:
            self.xerr = np.core.numeric.asarray(xerr.values) + 0.0
        except:
            if xerr is not None:
                self.xerr = np.core.numeric.asarray(xerr)
            else:
                self.xerr = None
        # Aux data
        self.name = name
        self.verbose = verbose
        # Initialize fit results
        self.clear_fit()

    def clear_fit(self):
        """Clear the fit data (not the input data!)"""
        # Disable fit flag
        self.fitted = False
        # Model data (general)
        self._model = 'NOSTILLSET'
        self.x = None
        self.y = None
        self.w = None
        self.errfun = None
        self.synthf = None
        self.loglog = None
        self.dof = None
        self.pcorr = None
        # Model data (needed to the fit/s)
        self.p0 = None
        # Model data (resulting from the fit/s)
        self.pfit = None
        self.pfitErr2 = None
        self.amp = None
        self.index = None
        self.ampErr = None
        self.indexErr = None
        self.RMSE = None
        self.logLik0 = None
        self.logLik = None
        self.R2 = None
        self.AICc = None
        self.resids = None
        self.errors = None
        # Model data (specific to averaged models)
        self.w_norm = None
        self.w_logn = None

    def powerlaw(self, x, amp, index, cte=0.0):  # Data usually in lin scale
        """Define a power-law with given amplitude and index."""
        return amp * (x ** index) + cte

    def fitlinfunc(self, p, x):  # Data usually entered in log scale
        """Aux for fitting with errlinfunc: lineal fit."""
        return p[0] + p[1] * x  # + np.log(1 + p[2]/(np.exp(p[0] + p[1] * x)))

    def errlinfunc(self, p, x, y, w=None):  # Data usually entered in log scale
        """Error function of a lineal fit."""
        return (y - self.fitlinfunc(p, x)) if w is None else (
            w * y - w * self.fitlinfunc(p, x))

    def linerrlogfit(self, p, x, y, w=None):  # Data usually in log scale
        """Error in lineal scale of a loglog fit (data in log scale)."""
        return (np.exp(y) - np.exp(self.fitlinfunc(p, x))) if w is None else (
            w * np.exp(y) - w * np.exp(self.fitlinfunc(p, x)))

    def synthlinfunc(self, p, x, s, r, w=None):  # Data usually in log scale
        """Error function of a lineal fit with synthetic y-data."""
        return (
            (s[0] + s[1] * x + r) - self.fitlinfunc(p, x)
        ) if w is None else (
            w * (s[0] + s[1] * x + r) - w * self.fitlinfunc(p, x)
        )

    def fitpowfunc(self, p, x):  # Data usually entered in lin scale
        """Aux for fitting with errpowfunc: non-lineal power-law fit."""
        return p[0] * (x ** p[1])  # + p[2]

    def errpowfunc(self, p, x, y, w=None):  # Data usually entered in lin scale
        """Error function of a non-lineal power-law fit."""
        return (y - self.fitpowfunc(p, x)) if w is None else (
            w * y - w * self.fitpowfunc(p, x))

    def synthpowfunc(self, p, x, s, r, w=None):  # Data usually in lin scale
        """Error function of a non-lineal power-law fit with synthetic y-data.
        """
        return (
            (s[0] * (x ** s[1]) + r) - self.fitpowfunc(p, x)
        ) if w is None else (
            w * (s[0] * (x ** s[1]) + r) - w * self.fitpowfunc(p, x))

    def copy(self):
        """Copy source data."""
        return(PowerFitModel(
            x=self.xdata, y=self.ydata, xerr=self.xerr,
            name=self.name, verbose=self.verbose)
        )

    def get(self, full=False):
        """Retrieves the results from the fitted model."""
        if self.fitted:
            if full:
                return({
                    "amp": self.amp, "ampErr": self.ampErr,
                    "index": self.index, "indexErr": self.indexErr,
                    "R2": self.R2, "pcorr2": self.pcorr ** 2,
                    "model": self.getModel(),
                    "AICc": self.AICc, "RMSE": self.RMSE,
                    "logLikDoF": self.logLik / self.dof
                })
            else:
                return(
                    self.amp, self.ampErr, self.index, self.indexErr,
                    self.R2, self.pcorr ** 2, self.getModel()
                )
        else:
            raise UnfittedModelError(self, sys._getframe())

    def fit_func(self, x):
        """From x-data, return fitted y-data depending on the model."""
        if self.fitted:
            if (self.isModel('LLR') or
                    self.isModel('xWLLR') or self.isModel('IWLLR')):
                return(self.fitlinfunc(self.pfit, x))
            elif (self.isModel('NLR') or
                    self.isModel('xWNLR') or self.isModel('IWNLR')):
                return(self.fitpowfunc(self.pfit, x))
            elif self.isModel('AVG') or self.isModel('xWAVG'):
                return(self.w_norm * self.fitpowfunc(self.pfit, x) +
                       self.w_logn * self.fitlinfunc(self.pfit, x))
            else:
                raise UnknownModelError(self, sys._getframe(), self.getModel())
        else:
            raise UnfittedModelError(self, sys._getframe())

    def err_func(self, y=None, x=None, p=None, resids='native'):
        """Error function of the fit, depending on the model."""
        if p is None:
            p = self.pfit
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        w = None
        if self.w is not None:
            w = self.w
        if self.fitted:
            try:
                plst = p.tolist()
            except:
                try:
                    plst = list(p)
                except TypeError:
                    raise MissingParamsError(self, sys._getframe())
            if None in plst:
                raise MissingParamsError(self, sys._getframe())
            if (self.isModel('LLR') or
                    self.isModel('xWLLR') or self.isModel('IWLLR')):
                if resids == 'native':
                    return(self.errlinfunc(p, x, y, w=w))
                elif resids == 'lineal':
                    w = None
                    if self.xerr is not None:
                        w = 1.0 / self.xerr
                        # w /= w.sum() # Weights normalization
                    return(self.linerrlogfit(p, x, y, w=w))  # w=w
                else:
                    raise UnknownResidsError(self, sys._getframe(), resids)
            elif (self.isModel('NLR') or self.isModel('xWboot') or
                    self.isModel('xWNLR') or self.isModel('IWNLR')):
                return(self.errpowfunc(p, x, y, w=w))  # w=w
            elif self.isModel('AVG'):
                if resids == 'native':
                    return(
                        self.w_norm * self.errpowfunc(p, x, y) +
                        self.w_logn * self.errlinfunc(
                            [np.log(p[0]), p[1]], np.log(x), np.log(y)
                        )
                    )
                elif resids == 'lineal':
                    return(
                        self.w_norm * self.errpowfunc(p, x, y) +
                        self.w_logn * self.linerrlogfit(
                            [np.log(p[0]), p[1]], np.log(x), np.log(y)
                        )
                    )
                else:
                    raise UnknownResidsError(self, sys._getframe(), resids)
            elif self.isModel('xWAVG'):
                pfit = np.ndarray(2)
                pfit[1] = 1.0 / p[1]
                pfit[0] = p[0] ** (-pfit[1])
                pfit_pow = [pfit[0], pfit[1]]
                w_pow = 1.0 / self.xerr
                # w_pow /= w_pow.sum() # Weights normalization
                pfit[0] = -np.log(p[0]) * pfit[1]
                pfit_lin = [pfit[0], pfit[1]]
                w_lin = self.xdata / self.xerr
                # w_lin /= w_lin.sum() # Weights normalization
                if resids == 'native':
                    return(
                        self.w_norm * self.errpowfunc(pfit_pow,
                                                      x, y, w=w_pow) +
                        self.w_logn * self.errlinfunc(
                            pfit_lin, np.log(x), np.log(y), w=w_lin
                        )
                    )
                elif resids == 'lineal':
                    return(
                        self.w_norm * self.errpowfunc(
                            pfit_pow, x, y, w=None  # w=w_pow
                        ) +
                        self.w_logn * self.linerrlogfit(
                            pfit_lin, np.log(x), np.log(y), w=None  # w=w_pow
                        )
                    )
                else:
                    raise UnknownResidsError(self, sys._getframe(), resids)
            else:
                raise UnknownModelError(self, sys._getframe(), self.getModel())
        else:
            raise UnfittedModelError(self, sys._getframe())

    def pwr_law(self, x):
        """From x-data, return power-law y-data for a fitted model."""
        if self.fitted:
            return(self.powerlaw(x, self.amp, self.index))
        else:
            raise UnfittedModelError(self, sys._getframe())

    def setModel(self, model):
        """Set the type of the model."""
        if not self.fitted:
            if self.pfit is not None:
                if model in self.supported:
                    self._model = model
                else:
                    raise UnknownModelError(self, sys._getframe(), model)
                self.fitted = True
            else:
                raise MissingParamsError(self, sys._getframe())
        else:
            raise ModelError(
                '\nERROR! PowerFitModel.setModel():\n\t' +
                'Cannot set model for an already fitted model (' +
                self.getModel() + ')!'
            )

    def getModel(self):
        """Get the type of the model."""
        if self.fitted:
            return(self._model)
        else:
            raise UnfittedModelError(self, sys._getframe())

    def isModel(self, model):
        """Check if the model is of the specified type."""
        if self.fitted:
            if model in self.supported:
                return(self._model == model)
            else:
                raise UnknownModelError(self, sys._getframe(), model)
        else:
            raise UnfittedModelError(self, sys._getframe())

    def setupTest(
            self, errormodel, amp=0.5,
            index=0.5, scale=5, points=20,
            bxerr=False
    ):
        """Setup test mode.

        TO DO: Check and fix xerr array generation!
        """
        # Set the error model (no the fit model!)
        try:
            errormodel = errormodel[:3].lower()
        except:
            raise UnknownErrorModelError(self, sys._getframe(), errormodel)
        self.fitted = False
        # Set new parameters
        self.ampTest = amp
        self.indexTest = index
        # Generate the test data
        self.xdata = np.logspace(-6.5, -0.5, num=points)
        self.xdata /= self.xdata.sum()
        if errormodel == 'non':  # No Noise
            self.name = 'ccTEST_PowerFit_NoNoise'
            self.ydata = self.powerlaw(self.xdata, self.ampTest,
                                       self.indexTest)
        elif errormodel == 'add':  # Additive gaussian noise -> absolute value!
            self.name = 'ccTEST_PowerFit_AdditNoise'
            self.ydata = abs(np.array([
                self.powerlaw(i, self.ampTest, self.indexTest) +
                np.random.normal(0.0, 1e-2 / scale, 1)[0] for i in self.xdata
            ]))
        elif errormodel == 'mul':  # Multiplicative gaussian noise
            self.name = 'ccTEST_PowerFit_MultiNoise'
            self.ydata = np.array([
                self.powerlaw(i, self.ampTest, self.indexTest) +
                np.random.normal(
                    0.0,
                    self.powerlaw(i, self.ampTest, self.indexTest) / scale,
                    1
                )[0] for i in self.xdata
            ])
        elif errormodel == 'mix':  # Mixed additive and multip. gaussian noise
            self.name = 'ccTEST_PowerFit_MixedNoise'
            if (amp - 0.5) > _EPS or (index - 0.5) > _EPS:
                print(
                    'WARNING! In FitPowPlot.setupTest(): ' +
                    'With "mix(ed)" errormodel both "amp" and "index"' +
                    'are overriden and set to 0.5'
                )
            self.xdata = np.logspace(-5.5, -0.5, num=20)  # Fix to 20 points!!!
            self.xdata /= self.xdata.sum()
            self.ydata = np.array([
                0.0015589, 0.00143023, 0.00167226, 0.00287218, 0.00382428,
                0.00454071, 0.00677716, 0.00866711, 0.01206321, 0.01638679,
                0.02194541, 0.02969171, 0.03915349, 0.05544938, 0.07400991,
                0.10126725, 0.13293193, 0.1866955, 0.2424437, 0.33900444
            ])
            # self.ydataTest=np.array([
            #    0.00134353,0.00264097,0.00480968,0.00934596,0.01733283,
            #    0.0329942,0.06220157,0.11826925,0.22420283,0.4245471
            # ]) # For R comparison
        elif errormodel == 'bad':  # No powerlaw data
            self.name = 'ccTEST_PowerFit_BadPowerLaw'
            self.ydata = np.exp(self.xdata)
        else:
            raise UnknownErrorModelError(self, sys._getframe(), errormodel)
        # Generate the error in x
        if bxerr:
            self.xerr = 1e-8 + 2e-1 * self.xdata
            # self.xerr = np.ones(len(self.xdata))
            # self.xerr = self.ydata / np.sqrt(9) # Suppose 9 times
        else:
            self.xerr = None

    def fit(self, model='LLR', p0=None):
        """Non-linear Least Squares Fit (by scipy.optimize.leastsq)."""
        # Aux recursive function to get the initial fit array
        def getp0(loglog):
            if loglog:
                model = 'LLR'
            else:
                model = 'NLR'
            if self.verbose:
                print('\n=*=*= ccData.PowerFitModel.fit.getp0 =*=*=\n Next ' +
                      model +
                      ' unweighted fit estimates p0 for the weighted one.')
            self.fit(model)
            p0 = [self.amp, self.index]
            self.clear_fit()
            return(p0)

        # Prepare data
        self.clear_fit()
        if model == 'LLR':
            self.loglog = True
            if p0 is None:
                p0 = [0.5, 0.75]
            self.p0 = [np.log(p0[0]), p0[1]]
            self.x = np.log(self.xdata)
            self.y = np.log(self.ydata)
            self.errfun = self.errlinfunc
        elif model == 'NLR':
            self.loglog = False
            if p0 is None:
                p0 = [0.5, 0.75]
            self.p0 = p0
            self.x = self.xdata.copy()
            self.y = self.ydata.copy()
            self.errfun = self.errpowfunc
        elif model == 'xWLLR':
            self.loglog = True
            if p0 is None:
                p0 = getp0(self.loglog)
            self.p0 = [-np.log(p0[0]) / p0[1], 1.0 / p0[1]]
            self.x = np.log(self.ydata)
            self.y = np.log(self.xdata)
            self.w = (self.xdata / self.xerr)
            # self.w /= self.w.sum() # Weights normalization
            self.errfun = self.errlinfunc
        elif model == 'xWNLR':
            self.loglog = False
            if p0 is None:
                p0 = getp0(self.loglog)
            self.p0 = [(1.0 / p0[0]) ** (1.0 / p0[1]), 1.0 / p0[1]]
            self.x = self.ydata.copy()
            self.y = self.xdata.copy()
            self.w = 1.0 / self.xerr
            # self.w /= self.w.sum() # Weights normalization
            self.errfun = self.errpowfunc
        else:
            raise UnsupportedModelError(self, sys._getframe(), model)

        # Aux data for calcs
        nobs = len(self.y)
        nparm = len(self.p0)
        self.dof = nobs - nparm

        # calcule fit and derived parameters
        self.pfit, pcov, infodict, errmsg, success = optimize.leastsq(
            self.errfun,
            self.p0,
            args=(self.x, self.y, self.w),
            full_output=True
        )

        # Set the model (and so, the fitted flag)
        self.setModel(model)

        # Get the sum of squares of residuals (RSS)
        self.resids = infodict['fvec']
        RSS = np.sum(self.resids ** 2)
        if (RSS < _EPS):
            RSS = 0.0

        # Get the array of errors (resids in the lineal scale)
        if self.loglog:
            self.errors = self.err_func(resids='lineal')
        else:
            self.errors = self.resids

        # The covariance returned is the reduced covariance or fractional cov.,
        # so one can multiply it by the reduced chi squared, s_sq. The errors
        # in the parameters are then the square root of the diagonal elements.
        if (self.dof > 0) and pcov is not None:
            # cov from R: As obtained by R generic function vcov(model), in
            #  order to keep the compatibility against R tests:
            MSE = RSS / self.dof  # Mean Squared Error
            self.RMSE = np.sqrt(MSE)  # Root Mean Squared Error
            cov = MSE * pcov
            # cov from NumPy: Some literature ignores the extra -2.0 factor in
            #  the denominator of the factor, but it is included here because
            #  the covariance of Multivariate Student-T (which is implied by a
            #  Bayesian uncertainty analysis) includes it. Plus, it gives a
            #  slightly more conservative estimate of uncertainty:
            # fac = RSS / (self.dof - 2.0)
            # cov = fac * pcov
            #
            self.pfitErr2 = np.diag(cov)
            #  Get parameter standard errors
            # parmSE = np.diag(np.sqrt(cov))
            #  Calculate the t-values
            # tvals = self.pfit / parmSE
            #  Get p-values
            # pvals = (1 - stats.t.cdf(np.abs(tvals), self.dof))*2
        else:
            self.RMSE = MSE = cov = np.inf

        # Calculate Pearson correlation coefficient between x and y
        try:
            pcorrden = np.sqrt(cov[0, 0] * cov[1, 1])
        except (TypeError):
            raise CovMatrixError(self, sys._getframe(), model, cov)
        if pcorrden < _EPS:
            self.pcorr = 0.0
        else:
            if abs(cov[0, 1] - pcorrden) > _EPS:
                self.pcorr = cov[0, 1] / pcorrden
            else:
                self.pcorr = 1.0
        # test for numerical error propagation
        if (self.pcorr > 1.0):
            self.pcorr = 1.0
        elif (self.pcorr < -1.0):
            self.pcorr = -1.0

        # Get Pearson corr. coefficient between real-y {y} and estimated-y {f}
        f = self.fit_func(self.x)
        corrcoef = np.corrcoef(self.y, f, ddof=0)[0, 1]  # Dif. ddof no effect!

        # Call fit2params to get final index and amplitude and associate errors
        self.fit2par()

        # Calculate log-likelihood and R^2
        R2 = self.fit2lllR2()
        if self.R2 < -1.0:
            self.R2 = corrcoef ** 2

        # Get AIC(c) (add 1 to the df to account for estimation of std error)
        k = 3  # Number of parameters of model
        AIC = -2 * self.logLik + k * (nobs + 1)
        self.AICc = 2 * k - 2 * self.logLik + 2 * k * (k + 1) / (nobs - k - 1)

        # Verbose output
        if self.verbose:
            print('\n***ccData.PowerFitModel.fit*** MODEL =>', model)
            print(' Array of initial values=', self.p0)
            print(' Array of fitted values =', self.pfit)
            print(' Diag covariance matrix =', self.pfitErr2)
            print(' Fit params squared corr coef(p[0],p[1]) =',
                  self.pcorr ** 2)
            # print(' Scaled covariance matrix:\n',cov)
            print(' V   =', self.amp, ' +/- ', self.ampErr)
            print(' beta=', self.index, ' +/- ', self.indexErr)
            print(' RMSE =', self.RMSE, '\tESS(lineal) =',
                  (self.errors ** 2).sum())
            print(' RSS(routine) =', RSS,
                  '\tRSS(native) =',
                  np.sum(self.err_func(resids='native') ** 2))
            # print(' tvals=', tvals,'\tpvals=',pvals)
            print(' logLikelihood(full-model) =', self.logLik)
            print(' logLikelihood(null-model) =', self.logLik0)
            print(' AIC =', AIC, '\tAICc =', self.AICc)
            # print(' leastsq suc =', success, '\terrm =', errmsg)
            print(' Squared Pearson corr coef(x,y) =',
                  (np.corrcoef(self.x, self.y, ddof=0)[0, 1]) ** 2)
            print(' Squared Pearson corr coef(x,f) =',
                  (np.corrcoef(self.x, f, ddof=0)[0, 1]) ** 2)
            print(' Squared Pearson corr coef(y,f) =', corrcoef ** 2)
            print(' R^2  =', R2, '\t R^2(generalized) =', self.R2)

    def fit2par(self):
        """Calculate final index and amplitude and associate errors."""
        if not self.fitted:
            raise UnfittedModelError(self, sys._getframe())
        if self.isModel('LLR'):
            self.index = self.pfit[1]
            self.indexErr = np.sqrt(self.pfitErr2[1])
            self.amp = np.exp(self.pfit[0])
            self.ampErr = np.sqrt(self.pfitErr2[0]) * self.amp
        elif self.isModel('NLR'):
            self.index = self.pfit[1]
            self.indexErr = np.sqrt(self.pfitErr2[1])
            self.amp = self.pfit[0]
            self.ampErr = np.sqrt(self.pfitErr2[0])
        elif self.isModel('xWLLR') or self.isModel('IWLLR'):
            self.index = 1.0 / self.pfit[1]
            self.amp = np.exp(-self.pfit[0] / self.pfit[1])
            self.indexErr = (np.sqrt(self.pfitErr2[1]) /
                             (self.pfit[1] * self.pfit[1]))
            self.ampErr = (np.exp(-self.pfit[0] / self.pfit[1]) /
                           self.pfit[1] * np.sqrt(
                self.pfitErr2[0] + self.pfit[0] * self.pfit[0] /
                (self.pfit[1] * self.pfit[1]) * self.pfitErr2[1]
                )
            )
        elif self.isModel('xWNLR') or self.isModel('IWNLR'):
            self.index = 1 / self.pfit[1]
            self.amp = self.pfit[0] ** (-1.0 / self.pfit[1])
            self.indexErr = (np.sqrt(self.pfitErr2[1]) /
                             (self.pfit[1] * self.pfit[1]))
            self.ampErr = (self.pfit[0] ** (-1 / self.pfit[1]) / self.pfit[1] *
                           np.sqrt(
                self.pfitErr2[0] / (self.pfit[0] * self.pfit[0]) +
                (np.log(self.pfit[0]) ** 2) /
                (self.pfit[1] * self.pfit[1]) * self.pfitErr2[1]
                )
            )
        elif self.isModel('AVG') or self.isModel('xWAVG'):
            # The same as case 'NLR' (see bootstrap function for the reason)
            self.index = self.pfit[1]
            self.indexErr = np.sqrt(self.pfitErr2[1])
            self.amp = self.pfit[0]
            self.ampErr = np.sqrt(self.pfitErr2[0])
        else:
            raise UnknownModelError(self, sys._getframe(), self.getModel())

    def fit2lllR2(self):
        """Calculate log-likelihood and generalized R^2."""
        if not self.fitted:
            raise UnfittedModelError(self, sys._getframe())
        # Do nothing for averaged model! Return the averaged R2 in bestFit()
        if self.isModel('AVG') or self.isModel('xWAVG'):
            return(self.R2)

        # Aux data for calcs
        nobs = len(self.y)

        # Get the sum of squares of residuals (RSS) or Residual Squared Sum
        RSS = np.sum(self.resids ** 2)  # = (errfunc(self.pfit, x, y)**2).sum()
        if (RSS < _EPS):
            RSS = 0.0
        # Get Total Squared Sum (TSS)= nobs * var_biased[y]
        TSS = 0.0
        if self.w is None:
            TSS = np.sum((self.y - np.mean(self.y)) ** 2)
        else:
            TSS = np.sum((self.w * self.y - self.w * np.mean(self.y)) ** 2)

        # Get biased variance and calculate log-likelihood
        #  of unrestricted and baseline/restricted/null(0) model
        s2b = RSS / nobs
        s2b0 = TSS / nobs
        # var = np.var(self.resids, ddof=1) # Variance of Residues (unbiased)
        self.logLik = self.logLik0 = 0.0
        if self.loglog:
            if (s2b > _EPS):
                self.logLik = (-nobs / 2 * np.log(2 * np.pi * s2b) -
                               RSS / (2 * s2b) - self.y.sum())
            else:
                self.logLik = np.inf
            self.logLik0 = (-nobs / 2 * np.log(2 * np.pi * s2b0) - nobs / 2 -
                            self.y.sum())
        else:
            if (s2b > _EPS):
                self.logLik = (-nobs / 2 * np.log(2 * np.pi * s2b) -
                               RSS / (2 * s2b))
            else:
                self.logLik = np.inf
            self.logLik0 = -nobs / 2 * np.log(2 * np.pi * s2b0) - nobs / 2

        # Calculate Coefficient of Determination
        # -Generalized expression
        self.R2 = 1 - np.exp(2 / nobs * (self.logLik0 - self.logLik))
        # -Traditional expression
        return(1 - RSS / TSS)

    @classmethod
    def bestFit(
            cls, x, y,
            xerr=None, p0=None,
            booting=False, forceboot=False,
            name='BestFitModel', verbose=False
    ):
        """Do best fit: Log-LR, Non-LR or model averaging.

        Follows the procedure in the Dokuwiki Power-law section.
        """
        # 1.a,b,c
        pfm_NLR = cls(
            x, y, xerr, name=name,
            verbose=(False if booting else verbose)
        )
        if xerr is None:
            pfm_NLR.fit(model='NLR', p0=p0)
        else:
            pfm_NLR.fit(model='xWNLR', p0=p0)
        # 1.d,e,f
        pfm_LLR = cls(
            x, y, xerr, name=name,
            verbose=(False if booting else verbose)
        )
        if xerr is None:
            pfm_LLR.fit(model='LLR', p0=p0)
        else:
            pfm_LLR.fit(model='xWLLR', p0=p0)
        # 2.
        pfm = None
        semilevelAICc = 8  # In the paper this is set to 2
        deltaAICc = pfm_NLR.AICc - pfm_LLR.AICc
        if (deltaAICc > semilevelAICc):  # or
            # xerr is not None): # CAUTION!!! TEMP WORKAROUND TO FORCE xWLLR!
            pfm = pfm_LLR
            pfm.w_norm = 0.0
            pfm.w_logn = 1.0
            del(pfm_NLR)
        elif (deltaAICc < -semilevelAICc):
            pfm = pfm_NLR
            pfm.w_norm = 1.0
            pfm.w_logn = 0.0
            del(pfm_LLR)
        else:
            # Create and populate an averaged-model PFM object
            pfm = cls(x, y, xerr, name=name, verbose=verbose)
            pfm.dof = len(pfm.ydata) - 2
            pfm.w_norm = 1 / (1 + np.exp(deltaAICc / 2))
            pfm.w_logn = 1 / (1 + np.exp(-deltaAICc / 2))
            if (np.isnan(pfm.w_norm) or np.isnan(pfm.w_logn)):
                pfm.w_norm = pfm.w_logn = 0.5
            pfm.amp = pfm_NLR.amp * pfm.w_norm + pfm_LLR.amp * pfm.w_logn
            pfm.index = pfm_NLR.index * pfm.w_norm + pfm_LLR.index * pfm.w_logn
            pfm.pfit = [pfm.amp, pfm.index]
            if xerr is None:
                pfm.x = pfm.xdata
                pfm.y = pfm.ydata
                pfm.setModel('AVG')
            else:
                pfm.x = pfm.ydata
                pfm.y = pfm.xdata
                pfm.setModel('xWAVG')
            if not booting:
                forceboot = True  # Ensure bootstrap as this is avrged. model
                # Parameters errors' worstcase estimation if bootstraping fails
                pfm.ampErr = pfm_NLR.ampErr + pfm_LLR.ampErr
                pfm.indexErr = pfm_NLR.indexErr + pfm_LLR.indexErr
                pfm.resids = pfm.err_func(resids='native')
                pfm.errors = pfm.err_func(resids='lineal')
                pfm.RMSE = np.sqrt((pfm.resids ** 2).sum() / pfm.dof)
            # WARNING! Estimation of coef. of determination by weighting
            pfm.R2 = pfm_NLR.R2 * pfm.w_norm + pfm_LLR.R2 * pfm.w_logn
            # WARNING! Estimation of fit-parameters correlation by weighting
            pfm.pcorr = pfm_NLR.pcorr * pfm.w_norm + pfm_LLR.pcorr * pfm.w_logn
            del(pfm_NLR)
            del(pfm_LLR)
        # Call the bootstrap routine (only mandatory for averaged models)
        if forceboot and not booting:
            try:
                pfm.bootstrap()  # pfm.bootstrapWild(dist='rademacher')
            except:
                print('WARNING! In PowerFitModel.bestFit(): ' +
                      'Call to bootstrap routine failed!')
                raise
        if verbose:
            if not booting:
                print('\n>>> ccData.PowerFitModel.bestFit <<<' +
                      '>>> Model used for the fit : ' + pfm.getModel())
                print('>>> AICc difference = ', deltaAICc)
                print('>>> w_norm = ', pfm.w_norm, '\tw_logn = ', pfm.w_logn)
                print('>>> V   =', pfm.amp, ' +/- ', pfm.ampErr)
                print('>>> beta=', pfm.index, ' +/- ', pfm.indexErr)
            else:
                # print('>*> ccData.PowerFitModel.bestFit(model:' +
                #    pfm.getModel()+') B = ',pfm.amp,'\tbeta = ',pfm.index)
                pass
        return (pfm)

    @classmethod
    def bestFit2(
            cls, x, y,
            xerr=None, p0=None,
            booting=False, forceboot=False,
            name='BestFitModel2', verbose=False,
            legacy=False
    ):
        """Do best fit version 2: Bootstrap if x-Weighted (xerr not None).

        TO DO: Update the procedure in the Dokuwiki Power-law section.
        """
        pfm = []
        if xerr is None or legacy:
            pfm = cls.bestFit(
                x, y, xerr, p0, booting, forceboot, name, verbose)
        else:
            ps = []  # List of arrays with the fit parameters
            R2s = []  # List of determination coef. of the fits
            pcorrs = []  # List of correlation between parameters of the fits
            relerr = pfitold = np.inf
            numstep = numNLR = numLLR = numAVG = numbad = 0
            # Error-guided number of steps of iterative fitting
            fitsperstep = 100
            maxsteps = 100
            reltol = 1e-4
            while (relerr > reltol and numstep < maxsteps):
                numstep += 1
                for i in range(fitsperstep):
                    rnd_x = np.random.normal(x, xerr)  # np.r.normal(mean,sdev)
                    # Filter values under EPS (avoid negatives!)
                    filter = rnd_x > _EPS
                    try:
                        pfm_rnd = cls.bestFit(
                            rnd_x[filter], y[filter], xerr=None,
                            booting=True, name=name, verbose=verbose
                        )
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        numbad += 1
                    else:
                        # As NLR model:
                        ps.append([pfm_rnd.amp, pfm_rnd.index])
                        R2s.append(pfm_rnd.R2)
                        pcorrs.append(pfm_rnd.pcorr)
                        if verbose:
                            if pfm_rnd.isModel('NLR'):
                                numNLR += 1
                            elif pfm_rnd.isModel('LLR'):
                                numLLR += 1
                            elif pfm_rnd.isModel('AVG'):
                                numAVG += 1
                            else:
                                numbad += 1
                        del(pfm_rnd)
                npps = np.array(ps)
                pfit = np.mean(npps, 0)
                # WARNING! Estimation of coef. of determination as an average!
                R2 = np.mean(np.array(R2s))
                relerr = np.max(np.abs(pfit - pfitold) / pfit)
                pfitold = pfit  # Save pfit to compute relerr on next iteration
                if verbose:
                    print('Fit-xWboot-> Step', numstep, '(', fitsperstep,
                          'fits ): Fit =', pfit,
                          '\trelerr =', relerr, '\tR2 = ', R2)
            # Warn on max steps reached
            if numstep >= maxsteps:
                print('WARNING! xWboot reached max number of steps: ' +
                      'Results may be inaccurate!')
            # Create and populate an xWboot PFM object
            pfm = cls(x, y, xerr=xerr, name=name, verbose=verbose)
            pfm.dof = len(pfm.ydata) - 2
            pfitold = pfm.pfit if pfm.pfit is not None else 0
            pfm.pfit = pfit
            (pfm.amp, pfm.index) = pfm.pfit
            pfm.x = pfm.xdata
            pfm.y = pfm.ydata
            pfm.w = pfm.xerr
            pfm.setModel('xWboot')
            pfm.resids = pfm.err_func(resids='native')
            pfm.errors = pfm.err_func(resids='lineal')
            pfm.R2 = R2
            # WARNING! Estimation of fit params. correlation as an average!
            pfm.pcorr = np.mean(np.array(pcorrs))
            RSS = (pfm.resids ** 2).sum()
            pfm.RMSE = np.sqrt(RSS / pfm.dof)
            Nsigma = 1.0
            pfm.pfitErr2 = (Nsigma * np.std(npps, 0)) ** 2
            (pfm.ampErr, pfm.indexErr) = np.sqrt(pfm.pfitErr2)
            #  Uncomment next 2 lines (and packages) to plot scatter V vs beta
            # dftmp = DataFrame(npps, columns=['V', 'beta'])
            # scatter_matrix(dftmp, alpha=0.2, figsize=(4, 4), diagonal='kde')
            if verbose:
                print('\n>>> ccData.PowerFitModel.bestFit2 <<<' +
                      '>>> Fitted model : ' + pfm.getModel())
                print('>>> V   =', pfm.amp, ' +/- ', pfm.ampErr)
                print('>>> beta=', pfm.index, ' +/- ', pfm.indexErr)
                print('>>> Total models computed =', fitsperstep * numstep)
                print(' -> NLR models =', numNLR, ' LLR models =', numLLR)
                print(' -> AVG models =', numAVG, ' Failed models =', numbad)
        return (pfm)

    def fitInvW(self, loglog=True):
        """Inverted LS (Least Squares) Fit with errors in x."""
        # prepare data
        self.clear_fit()
        self.loglog = loglog
        if self.loglog:
            self.x = np.log(self.ydata)
            self.y = np.log(self.xdata)
            self.w = (self.xdata / self.xerr)
            self.errfun = self.errlinfunc
        else:
            self.x = self.ydata.copy()
            self.y = self.xdata.copy()
            self.w = 1.0 / self.xerr
            self.errfun = self.errpowfunc

        # set up our least squares problem
        nparm = 2
        nobs = len(self.y)
        self.dof = nobs - nparm
        rcond = len(self.x) * np.core.finfo(self.x.dtype).eps
        # set up least squares equation for powers of x
        lhs = np.lib.twodim_base.vander(self.x, nparm)
        rhs = self.y.copy()
        # apply weighting
        lhs *= self.w[:, np.core.numeric.newaxis]
        rhs *= self.w

        # scale lhs to improve condition number and solve
        scale = np.sqrt((lhs * lhs).sum(axis=0))
        lhs /= scale
        c, RSS, rank, s = np.linalg.lstsq(lhs, rhs, rcond)
        c = (c.T / scale).T  # broadcast scale coefficients

        # warn on rank reduction, which indicates an ill conditioned matrix
        if rank != nparm:
            print('WARNING! ccPlot.FitPowPlot.fitByInvW : ' +
                  'LeastSq fit may be poorly conditioned!')

        Vbase = np.linalg.inv(np.core.dot(lhs.T, lhs))
        Vbase /= np.core.numeric.outer(scale, scale)
        # Some literature ignores the extra -2.0 factor in the denominator, but
        #  it is included here because the covariance of Multivariate Student-T
        #  (which is implied by a Bayesian uncertainty analysis) includes it.
        #  Plus, it gives a slightly more conservative estimate of uncertainty.
        MSE = RSS / self.dof  # Mean Squared Error
        self.RMSE = np.sqrt(MSE)  # Root Mean Squared Error
        covar = MSE * Vbase

        # Calculate correlation coefficient
        try:
            pcorrden = np.sqrt(covar[0, 0] * covar[1, 1])
        except (TypeError):
            print('ERROR! In ccData.PowerFitModel.fitInvW' +
                  ', covar matrix is scalar:', covar)
            raise
        if pcorrden < _EPS:
            self.pcorr = 0.0
        else:
            if abs(covar[0, 1] - pcorrden) > _EPS:
                self.pcorr = (covar[0, 1] / pcorrden)
            else:
                self.pcorr = 1.0

        # Calculate the direct fit coeffs taking into account that
        # the polynomial coefficients are highest power first
        # NOTE: covar matrix has the arrangement of the diagonal
        #       elements opposite to the cov matrix of fit() method.
        index = c[0]
        indexErr = np.sqrt(covar[0, 0])
        amp = 0
        ampErr = 0
        if loglog:
            amp = np.exp(c[1])
            ampErr = np.sqrt(covar[1, 1]) * amp
        else:
            amp = c[1]
            ampErr = np.sqrt(covar[1, 1])

        # Calculate the inverted fit coeffs.
        self.pfit = c[::-1]
        self.pfitErr2 = np.diag(covar)[::-1]
        if self.loglog:
            self.setModel('IWLLR')
            self.index = 1 / c[0]
            self.amp = np.exp(-c[1] / c[0])
            c2_0 = c[0] * c[0]
            self.indexErr = indexErr / c2_0
            self.ampErr = (
                np.exp(-self.pfit[0] / self.pfit[1]) / self.pfit[1] *
                np.sqrt(
                    covar[1, 1] + self.pfit[0] * self.pfit[0] /
                    (self.pfit[1] * self.pfit[1]) * covar[0, 0]
                )
            )
        else:
            self.pfit = c[::-1]
            self.setModel('IWNLR')
            self.index = 1 / index
            self.amp = np.exp(-c[1] / c[0])
            self.indexErr = indexErr
            self.ampErr = ampErr

        # Get the arrays of resids and errors (resids in the lineal scale)
        self.resids = self.err_func(resids='native')
        if self.loglog:
            self.errors = self.err_func(resids='lineal')
        else:
            self.errors = self.resids

        # Get Pearson corr. coefficient between real-y {y} and estimated-y {f}
        f = self.fit_func(self.x)
        corrcoef = np.corrcoef(self.y, f, ddof=0)[0, 1]  # Dif. ddof no effect!
        self.R2 = corrcoef ** 2

        if self.verbose:
            print('\n***ccData.PowerFitModel.fitInvW*** MODEL =>',
                  self.getModel())
            print(' Array of fitted values [::-1] =', self.pfit)
            print(' Diag covariance matrix [::-1] =', self.pfitErr2)
            print(' Fit params squared corr coef(p[0],p[1]) =',
                  self.pcorr ** 2)
            print(' V   =', self.amp, ' +/- ', self.ampErr)
            print(' beta=', self.index, ' +/- ', self.indexErr)
            print(' RMSE =', self.RMSE, '\tESS(lineal) =',
                  (self.errors ** 2).sum())
            print(' RSS(routine) =', RSS,
                  '\tRSS(calc) =', (self.resids ** 2).sum())
            print(' Squared Pearson corr coef(x,y) =',
                  (np.corrcoef(self.x, self.y, ddof=0)[0, 1]) ** 2)
            print(' Squared Pearson corr coef(x,f) =',
                  (np.corrcoef(self.x, f, ddof=0)[0, 1]) ** 2)
            print(' Squared Pearson corr coef(y,f) =', self.R2)

    def prepostbootstrap(self, prepost):
        """Pre and post for any bootstrap routine."""
        if not self.fitted:
            raise UnfittedModelError(self, sys._getframe())
        # Fixed parameters
        fitsperstep = min(len(self.ydata) * 10, 100000)
        steps = 1
        # Bootstrap preprocessing
        if prepost == 'pre':
            if self.resids is None:
                self.resids = self.err_func(resids='native')
            if self.errors is None:
                self.errors = self.err_func(resids='lineal')
            if self.verbose:
                print('\nB->ccData.PowerFitModel.prepostbootstrap()<-| ' +
                      'MODEL =>', self.getModel())
                print('B-> Starting fit array =', self.pfit, '  RSS = ',
                      (self.resids ** 2).sum(), '\tESS = ',
                      (self.errors ** 2).sum())
            return(
                self.x, self.y, self.w,
                self.errfun, self.synthf, self.loglog,
                fitsperstep, steps
            )
        # Bootstrap postprocessing
        elif prepost == 'post':
            # Call fit2par to get final index and amplitude and their errors
            self.fit2par()
            # Calculate log-likelihood and R^2
            R2 = self.fit2lllR2()
            if self.R2 < -1.0:
                self.R2 = np.corrcoef(
                    self.y, self.fit_func(self.x), ddof=0
                )[0, 1] ** 2
            # Verbose output
            if self.verbose:
                print('B-> Array of fitted values =', self.pfit)
                print('B-> Diag covariance matrix =', self.pfitErr2)
                print('B-> V   =', self.amp, ' +/- ', self.ampErr)
                print('B-> beta=', self.index, ' +/- ', self.indexErr)
                print('B-> RMSE =', self.RMSE, '\tESS =',
                      (self.errors ** 2).sum())
                print('B-> logLikelihood(full-model) =', self.logLik)
                print('B-> logLikelihood(null-model) =', self.logLik0)
                print('B-> R^2  =', R2, '\t R^2(generalized) =', self.R2)
        else:
            raise Exception('ERROR! In PowerFitModel.prepostboot(): ' +
                            'invalid prepost \'' + str(prepost) +
                            '\' (options: \'pre\', \'post\')')

    def bootstrap(self):
        """
        Estimate the confidence interval of the fitted parameters
        using the ordinary bootstrap method.
        """
        # Get data adapted to model
        (x, y, w,
         errfun, synthf, loglog,
         fitsperstep, steps) = self.prepostbootstrap('pre')
        # Aux data for calcs
        nobs = len(y)
        ps = []
        # Variable number of steps of iterative fitting
        for step in range(steps):
            for i in range(fitsperstep):
                # xboot = np.concatenate((x[:i],x[i+1:]))
                rnd_index = np.random.random_integers(0, nobs - 1, nobs)
                if not self.isModel('AVG') and not self.isModel('xWAVG'):
                    rnd_fit, rnd_cov, infodict, errmsg, success = \
                        optimize.leastsq(
                            errfun,
                            self.pfit,
                            args=(
                                x[rnd_index],
                                y[rnd_index],
                                None if w is None else w[rnd_index]
                            ),
                            full_output=True
                        )
                    if success in [1, 2, 3, 4]:
                        ps.append(rnd_fit)
                else:
                    pfm_rnd = self.bestFit(
                        self.xdata[rnd_index], self.ydata[rnd_index],
                        xerr=(
                            None if self.xerr is None else self.xerr[rnd_index]
                        ),
                        p0=[self.amp, self.index], booting=True,
                        name=self.name, verbose=self.verbose
                    )
                    # Next AVG and xWAVG as NLR model:
                    ps.append([pfm_rnd.amp, pfm_rnd.index])
                    del(pfm_rnd)
            npps = np.array(ps)
            self.pfit = np.mean(npps, 0)
            self.resids = self.err_func(resids='native')
            self.errors = self.err_func(resids='lineal')
            RSS = (self.resids ** 2).sum()
            if self.verbose:
                print('B--> Step', step + 1, '(', fitsperstep,
                      'fits ): Fit array =', self.pfit, '\tRSS = ', RSS)

        self.RMSE = np.sqrt(RSS / self.dof)
        Nsigma = 1.0
        self.pfitErr2 = (Nsigma * np.std(npps, 0)) ** 2
        # Bootstrap postprocessing
        self.prepostbootstrap('post')

    def bootstrapRR(self):
        """
        Estimate the confidence interval of the fitted parameters
        using the 'Resampling Residuals' bootstrap method.
        """
        # Get data adapted to model
        (x, y, w,
         errfun, synthf, loglog,
         fitsperstep, steps) = self.prepostbootstrap('pre')
        # Aux data for calcs
        nobs = len(y)
        ps = []
        # Variable number of steps of iterative fitting
        for step in range(steps):
            for i in range(fitsperstep):
                rnd_index = np.random.random_integers(0, nobs - 1, nobs)
                rnd_fit, rnd_cov, infodict, errmsg, success = \
                    optimize.leastsq(
                        synthf,
                        self.pfit,
                        args=(x, self.pfit, self.resids[rnd_index]),
                        full_output=True
                    )
                if success in [1, 2, 3, 4]:
                    ps.append(rnd_fit)
            npps = np.array(ps)
            self.pfit = np.mean(npps, 0)
            self.resids = self.err_func(resids='native')
            self.errors = self.err_func(resids='lineal')
            RSS = (self.resids ** 2).sum()
            if self.verbose:
                print('RR-> Step', step + 1, '(', fitsperstep,
                      'fits ): Fit array =', self.pfit, '\tRSS = ', RSS)

        self.RMSE = np.sqrt(RSS / self.dof)
        Nsigma = 1.0
        self.pfitErr2 = (Nsigma * np.std(ps, 0)) ** 2
        # Bootstrap postprocessing
        self.prepostbootstrap('post')

    def bootstrapWild(self, dist='normal'):
        """
        Estimate the CIs of the fitted parameters using the 'Wild'
        bootstrap method (Wu, 1986), that is suited when the model
        exhibits heteroskedasticity and for smaller sample sizes.
        """
        if dist.lower() == 'normal':
            normal = True
        elif dist.lower() == 'rademacher':
            normal = False
        else:
            raise Exception('ERROR! In PowerFitModel.bootstrapWild(): ' +
                            'invalid dist \'' + dist +
                            '\' (options: \'normal\', \'rademacher\')')
        # Get data adapted to model
        (x, y, w,
         errfun, synthf, loglog,
         fitsperstep, steps) = self.prepostbootstrap('pre')
        # Aux data for calcs
        nobs = len(y)
        ps = []
        # Variable number of steps of iterative fitting
        for step in range(steps):
            for i in range(fitsperstep):
                if normal:
                    rnd_delta = np.random.normal(0.0, 1.0, nobs)
                else:
                    rnd_delta = np.random.random_integers(0, 1, nobs) * 2 - 1
                rnd_resids = self.resids * rnd_delta
                rnd_fit, rnd_cov, infodict, errmsg, success = \
                    optimize.leastsq(
                        synthf,
                        self.pfit,
                        args=(
                            (x, self.pfit, rnd_resids) if w is None else
                            (x, self.pfit, rnd_resids, w)
                        ),
                        full_output=True
                    )
                if success in [1, 2, 3, 4]:
                    ps.append(rnd_fit)
            npps = np.array(ps)
            self.pfit = np.mean(npps, 0)
            self.resids = self.err_func(resids='native')
            self.errors = self.err_func(resids='lineal')
            RSS = (self.resids ** 2).sum()
            if self.verbose:
                print('Wld-' + ('N' if normal else 'R') + '> Step', step + 1,
                      '(', fitsperstep, 'fits ): Fit array =', self.pfit,
                      '\tRSS = ', RSS)

        self.RMSE = np.sqrt(RSS / self.dof)
        Nsigma = 1.0
        self.pfitErr2 = (Nsigma * np.std(ps, 0)) ** 2
        # Bootstrap postprocessing
        self.prepostbootstrap('post')

    def bootstrapMC(self):
        """
        Estimate the confidence interval of the fitted parameters
        using the bootstrap Monte-Carlo method
        http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
        """
        # Get data adapted to model
        (x, y, w,
         errfun, synthf, loglog,
         fitsperstep, steps) = self.prepostbootstrap('pre')
        # Aux data for calcs
        nobs = len(y)
        ps = []
        # Variable number of steps of iterative fitting
        for step in range(steps):
            s_resids = np.std(self.resids)
            # Random data sets per step are generated and fitted
            for i in range(fitsperstep):
                randomDelta = np.random.normal(0.0, s_resids, nobs)
                randomYdata = y + randomDelta
                rnd_fit, rnd_cov, infodict, errmsg, success = \
                    optimize.leastsq(
                        errfun,
                        self.pfit,
                        args=(
                            (x, randomYdata, None) if w is None else
                            (x, randomYdata, w)
                        ),
                        full_output=True)
                if success in [1, 2, 3, 4]:
                    ps.append(rnd_fit)
            npps = np.array(ps)
            self.pfit = np.mean(npps, 0)
            self.resids = self.err_func(resids='native')
            self.errors = self.err_func(resids='lineal')
            RSS = (self.resids ** 2).sum()
            if self.verbose:
                print('MC-> Step', step + 1, '(', fitsperstep,
                      'fits ): Fit array =', self.pfit, '\tRSS = ', RSS)

        self.RMSE = np.sqrt(RSS / self.dof)
        Nsigma = 1.0  # 1sigma gets approximately the same as methods above
        # 1sigma corresponds to 68.3% confidence interval
        # 2sigma corresponds to 95.44% confidence interval
        self.pfitErr2 = (Nsigma * np.std(ps, 0)) ** 2

        # Bootstrap postprocessing
        self.prepostbootstrap('post')
