from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy 

ext_modules = [
    Extension(
        "wftools",
        ["wftools.pyx"],
        #extra_compile_args = ["-O3", "-funroll-loops"],
    ),
    Extension(
        "tensorutils",
        ["tensorutils.pyx"],
    ),
#    Extension(
#        "sparsemat",
#        ["sparsemat.pyx"],
#        #extra_compile_args = ["-O3", "-funroll-loops"],
#        #extra_compile_args=['-fopenmp'],
#        #extra_link_args=['-fopenmp'],
#    )
    Extension(
        "sparsemat",
        ["sparsemat.pyx","spmv.cpp"],
        extra_compile_args = ["-lomp"],
        extra_link_args = ["-lomp"],
        language="c++",
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cyext',
    ext_modules=cythonize(ext_modules),
    include_dirs = [numpy.get_include(),scipy.get_include()]
)

#setup(
#    ext_modules = cythonize("sparsemat.pyx"),
#    include_dirs = [numpy.get_include()]
#)
