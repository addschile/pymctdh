pymctdh - an MCTDH code written entirely in python (and a little cython)

This code is in its very early stages and so it is not properly documented or
tested. Feel free to mess around with it as you please. If you find any bugs or
have any suggestions, I'd be happy to hear it!

I don't have a dedicated setup.py ready to go, so if you'd like to install
pymctdh you'll have to do the following:
- ensure that you have python3 (I don't think python2 is unusable, I just can't
guarantee it'll work)
- ensure that you have numpy, scipy, and cython (I have tested this code on
versions 1.17.2, 1.3.1, and 0.29.13, repsectively)
- after you've done these you're ready to install the cython code, starting from
the source directory do the following
- cd cy
- python setup.py build_ext --inplace
- the final step is to simply add the path of the pymctdh main directory to
your $PYTHONPATH environment variable

On the agenda of things to do or implement:
- setup.py
- Constant mean field integrator with fixed and adaptive timestep
- Computation of more complicated expectation values
- Spectroscopic simulations
- Improved relaxation
- Multilayer version of MCTDH
- projector splitting algorithm
