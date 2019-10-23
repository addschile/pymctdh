import numpy as np
import sys
sys.path.append('/Users/addisonschile/Software/pymctdh')
import units
from wavefunction import Wavefunction
from hamiltonian import Hamiltonian
from pbasis import PBasis
from qoperator import QOperator
from vmfpropagate import vmfpropagate,vmfpropagatejumps

if __name__ == "__main__":


    nel    = 2
    nmodes = 4
    nspfs = np.array([[8, 13, 7, 6],
                     [7, 12, 6, 5]], dtype=int)
    npbfs = np.array([22, 32, 21, 12])

    pbfs = list()
    pbfs.append( PBasis(['ho', 22, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 32, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 21, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 12, 1.0, 1.0]) )

    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    wf.generate_ic(1)

    w10a  =  0.09357
    w6a   =  0.0740
    w1    =  0.1273
    w9a   =  0.1568
    delta =  0.46165
    lamda =  0.1825
    k6a1  = -0.0964
    k6a2  =  0.1194
    k11   =  0.0470
    k12   =  0.2012
    k9a1  =  0.1594
    k9a2  =  0.0484
    gam   =  0.005*0.0367493

    # make hamiltonian
    hterms = []
    hterms.append({'coeff':   -delta, 'units': 'ev', 'elop': 'sz'}) # el only operator
    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  'KE'}) # mode 1 terms
    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': 'q^2'})
    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 1, 'ops':  'KE'}) # mode 2 terms
    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 1, 'ops': 'q^2'})
    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 2, 'ops':  'KE'}) # mode 3 terms
    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 2, 'ops': 'q^2'})
    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 3, 'ops':  'KE'}) # mode 4 terms
    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 3, 'ops': 'q^2'})
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': 'q'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 2 el 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 2, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 2, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 3, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 3, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 4 el 1
    #hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 0, 'elop':   '1', 'ops': 'adag*a'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 1, 'elop':   '1', 'ops': 'adag*a'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 2, 'elop':   '1', 'ops': 'adag*a'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 3, 'elop':   '1', 'ops': 'adag*a'}) # parameters for effective hamiltonian
    hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 0, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 1, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 2, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    hterms.append({'coeff': -gam*0.5j, 'units': 'au', 'modes': 3, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    dt = 0.1
    times = np.arange(0.0,120.,dt)*units.convert_to('fs')

    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4_nonhermitian_down.txt')
