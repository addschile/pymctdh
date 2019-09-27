from distutils.core import setup
from Cython.Build import cythonize
import numpy
import scipy

setup(
    ext_modules = cythonize("wftools.pyx"),
    include_dirs = [numpy.get_include(),scipy.get_include()]
)

