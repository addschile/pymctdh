import numpy as np
import sys
sys.path.append('/Users/addisonschile/Software/pymctdh')
import units
from wavefunction import Wavefunction
from hamiltonian import Hamiltonian
from pbasis import PBasis
from vmfpropagate import vmfpropagate

def make_a(npbf):
    a = np.zeros((npbf,)*2)
    for i in range(npbf-1):
        a[i,i+1] = np.sqrt(float(i+1))
    return a

def make_q(npbf):
    q = np.zeros((npbf,)*2)
    for i in range(npbf-1):
        q[i,i+1] = np.sqrt(0.5*float(i+1))
        q[i+1,i] = np.sqrt(0.5*float(i+1))
    return q

def make_ic(npbfs):
    # mode 1
    # 10a
    n10a = npbfs[0][0]
    w10a =  0.09357
    a    = make_a(n10a)
    h10a = w10a*(np.dot(a.T,a)+0.5*np.eye(n10a))
    # 6a
    n6a  = npbfs[0][1]
    w6a  =  0.0740
    k6a1 = -0.0964
    k6a2 =  0.1194
    a    = make_a(n6a)
    q    = make_q(n6a)
    h6a1 = w6a*(np.dot(a.T,a)+0.5*np.eye(n6a)) + k6a1*q
    h6a2 = w6a*(np.dot(a.T,a)+0.5*np.eye(n6a)) + k6a2*q
    # construct full hamiltonian and eigenstates
    # 1
    H11  = np.kron(h10a,np.eye(n6a))
    H11 += np.kron(np.eye(n10a),h6a1)
    w11,v11 = np.linalg.eigh(H11)
    # 2
    H12  = np.kron(h10a,np.eye(n6a))
    H12 += np.kron(np.eye(n10a),h6a2)
    w12,v12 = np.linalg.eigh(H12)

    # mode 2
    # 1
    n1  = npbfs[1][0]
    w1  =  0.1273
    k11 =  0.0470
    k12 =  0.2012
    a   = make_a(n1)
    q   = make_q(n1)
    h11 = w1*(np.dot(a.T,a)+0.5*np.eye(n1)) + k11*q
    h12 = w1*(np.dot(a.T,a)+0.5*np.eye(n1)) + k12*q
    # 9a
    n9a  = npbfs[1][1]
    w9a  =  0.1568
    k9a1 =  0.1594
    k9a2 =  0.0484
    a    = make_a(n9a)
    q    = make_q(n9a)
    h9a1 = w9a*(np.dot(a.T,a)+0.5*np.eye(n9a)) + k9a1*q
    h9a2 = w9a*(np.dot(a.T,a)+0.5*np.eye(n9a)) + k9a2*q
    # construct full hamiltonian and eigenstates
    # 1
    H21  = np.kron(h11,np.eye(n9a))
    H21 += np.kron(np.eye(n1),h9a1)
    w21,v21 = np.linalg.eigh(H21)
    # 2
    H22  = np.kron(h12,np.eye(n9a))
    H22 += np.kron(np.eye(n1),h9a2)
    w22,v22 = np.linalg.eigh(H22)

    return [[v11,v21],[v12,v22]]

if __name__ == "__main__":

    nel    = 2
    nmodes = 2
    nspfs = np.array([[6,6],
                      [4,4]], dtype=int)
    nspfs = np.array([[10,10],
                      [8,8]], dtype=int)
    #npbfs = [[22, 32],[21, 12]]
    #npbfs = [[12, 22],[12, 7]]
    npbfs = [[18, 28],[18, 14]]

    pbfs = list()
    pbfs.append( PBasis(['ho', npbfs[0], 1.0, 1.0, True],sparse=True) )
    pbfs.append( PBasis(['ho', npbfs[1], 1.0, 1.0, True],sparse=True) )

    vs = make_ic(npbfs)
    #npbfs = [[12, 22],[12, 7]]
    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    wf.generate_ic(1)
    for i in range(wf.nel):
        ind = wf.psistart[1,i]
        for j in range(wf.nmodes):
            v    = vs[i][j]
            nspf = wf.nspfs[i,j]
            npbf = wf.npbfs[j]
            ind += npbf
            for k in range(nspf-1):
                vtmp = v[k+1,:npbf]
                vtmp /= np.linalg.norm(vtmp)
                #print(npbf,vtmp.shape)
                wf.psi[ind:ind+npbf] = vtmp.copy()
                ind += npbf
    wf.orthonormalize_spfs()

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

    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4_profile_sparse_new_ic.txt')
