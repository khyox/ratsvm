"""
This is an object-oriented complex-systems analysis library.

A front-end interface is provided by the companion cmplxcruncher module,
which will import this library.

For the most part, direct use of the object-oriented library is encouraged when
programming; the front-end is useful for working with data, interactively or
in automatic mode (useful for processing several sources of data).

Modules included:

    :mod:`cmplxcruncher.config`
        provides constants and other package-wide stuff.
    :mod:`cmplxcruncher.models`
        defines the :class:`~cmplxcruncher.models.PowerFitModel` class.

    :mod:`cmplxcruncher.plots`
        defines the :class:`~cmplxcruncher.plots.CCplot` class, that is the
        base class for all the plots in the package.

    :mod:`cmplxcruncher.sessions`
        defines the :class:`~matplotlib.sessions.Session` base class for the
        three classes that represent the three preestablished modes of working:
        interactive (:class:`~matplotlib.sessions.InteractiveSession`),
        automatic (:class:`~matplotlib.sessions.AutomaticSession`),
        and test (:class:`~matplotlib.sessions.TestSession`).

cmplxcruncher was previously known as complexCruncher, and has been written by
JMMM in the DLS team of the University of Valencia (Spain).
"""

import sys
import distutils.version

__all__ = ["config", "models", "sessions", "plots"]

# cmplxcruncher release information
__version__ = '1.1rc12'
_verdata = 'Dec 2016'
_devflag = False

# required packages versions
_version_pandas = '0.16.0'
_version_matplotlib = '1.5.0'
_version_xlrd = '0.9'

# optional LaTeX extension (requires pylatex 1.0.0!!!)
_useLaTeX = True


def compare_versions(a, b):
    """Return True if a is greater than or equal to b."""
    if a:
        a = distutils.version.LooseVersion(a)
        b = distutils.version.LooseVersion(b)
        return a >= b
    else:
        return False

# python
major, minor1, minor2, s, tmp = sys.version_info
_pythonrel = (major == 2 and minor1 >= 7) or major >= 3
if not _pythonrel:
    raise ImportError('cmplxcruncher requires Python 2.7 or later')

# numpy
try:
    import numpy as np
except ImportError:
    raise ImportError("cmplxcruncher requires numpy")

_EPS = np.finfo(np.double).eps    # Define eps

# pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError("cmplxcruncher requires pandas")
else:
    if not compare_versions(pd.__version__, _version_pandas):
        raise ImportError(
            'pandas %s or later is required; you have %s' % (
                _version_pandas, pd.__version__))

# matplotlib
try:
    import matplotlib as mpl
except ImportError:
    raise ImportError("cmplxcruncher requires matplotlib")
else:
    if not compare_versions(mpl.__version__, _version_matplotlib):
        raise ImportError(
            'matplotlib %s or later is required; you have %s' % (
                _version_matplotlib, mpl.__version__))

# xlrd
try:
    import xlrd
except ImportError:
    raise ImportError("cmplxcruncher requires xlrd")
else:
    if not compare_versions(xlrd.__VERSION__, _version_xlrd):
        raise ImportError(
            'xlrd %s or later is required; you have %s' % (
                _version_xlrd, xlrd.__VERSION__))

# scipy
try:
    import scipy
except ImportError:
    raise ImportError("cmplxcruncher requires scipy")

# Optionals packages: pylatex and uncertainties
try:
    import pylatex
    from uncertainties import ufloat
except ImportError:
    _useLaTeX = False
