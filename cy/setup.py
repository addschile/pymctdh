from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy 

ext_modules = [
    Extension(
        "wftools",
        ["wftools.pyx"],
    ),
    Extension(
        "tensorutils",
        ["tensorutils.pyx"],
    ),
    Extension(
        "sparsemat",
        ["sparsemat.pyx"],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
    )
]

setup(
    name='sparsmat',
    ext_modules=cythonize(ext_modules),
    include_dirs = [numpy.get_include(),scipy.get_include()]
)

#setup(
#    ext_modules = cythonize("sparsemat.pyx"),
#    include_dirs = [numpy.get_include()]
#)
