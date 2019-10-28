import numpy as np
import sys
sys.path.append('/Users/addisonschile/Software/pymctdh')
import units
from wavefunction import Wavefunction
from hamiltonian import Hamiltonian
from pbasis import PBasis
from vmfpropagate import vmfpropagate

if __name__ == "__main__":

    nel    = 2
    nmodes = 2
    #nspfs = np.array([[6,6],
    #                  [4,4]], dtype=int)
    ##nspfs = np.array([[20,10],
    ##                  [20,10]], dtype=int)
    ##npbfs = [[22, 32],[21, 12]]
    #npbfs = [[12, 22],[12, 7]]
    nspfs = np.array([[10,10],
                      [8,8]], dtype=int)
    npbfs = [[18, 28],[18, 14]]

    pbfs = list()
    pbfs.append( PBasis(['ho', npbfs[0], 1.0, 1.0, True],sparse=True) )
    pbfs.append( PBasis(['ho', npbfs[1], 1.0, 1.0, True],sparse=True) )
    #pbfs.append( PBasis(['ho', npbfs[0], 1.0, 1.0, True]) )
    #pbfs.append( PBasis(['ho', npbfs[1], 1.0, 1.0, True]) )

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

    hterms = []
    hterms.append({'coeff':   -delta, 'units': 'ev', 'elop': 'sz'}) # el only operator
    # combined mode 0
    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 0, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 0, 'ops': '(1)*(q^2)'})
    # combined mode 1
    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 1, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 1, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 1, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 1, 'ops': '(1)*(q^2)'})
    # combined mode 0
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': '(q)*(1)'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 1
    # combined mode 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 1

    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    dt = 0.5
    times = np.arange(0.0,120.,dt)*units.convert_to('fs')

    #wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4_profile_sparse.txt')
    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4_profile_sparse_cpp.txt')
    #wf = vmfpropagate(times, ham, pbfs, wf, 'pyr_profile.txt')
