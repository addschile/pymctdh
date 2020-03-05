#!/usr/bin/env python
"""pymctdh
"""
import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'pymctdh'))  # to retrive _version

ext_modules = [
    Extension("pymctdh.cy.wftools",
        sources=["pymctdh/cy/wftools.pyx"],
        include_dirs = [numpy.get_include(),scipy.get_include()],
        extra_compile_args = ["-O3", "-funroll-loops"],
    ),
    Extension("pymctdh.cy.tensorutils",
        ["pymctdh/cy/tensorutils.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args = ["-O3", "-funroll-loops"],
    )
]

# Setup commands go here
setup(name='pymctdh',
      version='0.1',
      packages=['pymctdh','pymctdh/cy'],
      include_package_data=True,
      include_dirs = [numpy.get_include(),scipy.get_include()]
      ext_modules = cythonize(EXT_MODULES),
      cmdclass = {'build_ext': build_ext},
      author = 'Addison J. Schile',
      author_email = 'addschile@gmail.com',
      license = 'MIT',
      description = '',
      long_description = '',
      keywords = '',
      classifiers = CLASSIFIERS,
      platforms = PLATFORMS,
      requires = ['numpy (>=1.12)', 'scipy (>=1.0)', 'cython (>=0.21)'],
      package_data = PACKAGE_DATA,
      install_requires=['numpy>=1.12','scipy>=1.0', 'cython>=0.21'],
)

outmessage = """\
################################################################################
pymctdh is installed and ready to rumble
if you find this package useful please cite this github page and of course let
me know!
################################################################################
"""
print(outmessage)
