import numpy as np
from qutip import *

ev2au = 0.0367493
fs2au = 41.3413745758

def phi(n1,n2=None,nel=None):
    if nel==None:
        nel_ = 2
    else:
        nel_ = nel
    if n2==None:
        phiout = fock_dm(nel_,n1)
    else:
        phiout = basis(nel_,n1)*basis(nel_,n2).dag()
    return phiout

def eye(n, sparse=False):
    return qeye(n)

def construct_sys():

    # dimensions
    n10a = 25
    n6a  = 35
    n1   = 25
    n9a  = 20
    nstates = 2*n10a*n6a*n1*n9a
    # energies
    delta = 0.46165*ev2au
    # frequencies
    w10a = 0.09357*ev2au
    w6a  = 0.0740*ev2au
    w1   = 0.1273*ev2au
    w9a  = 0.1568*ev2au
    # holstein couplings
    # H_11
    k10a_1 = 0.0*ev2au
    k6a_1  = -0.0964*ev2au
    k1_1   = 0.0470*ev2au
    k9a_1  = 0.1594*ev2au
    # H_22
    k10a_2 = 0.0*ev2au
    k6a_2  = 0.1194*ev2au
    k1_2   = 0.2012*ev2au
    k9a_2  = 0.0484*ev2au
    # peierls coupling
    lamda = 0.1825*ev2au
    gam   = 0.005*ev2au
    
    # make annihilation operators
    a10a = destroy(n10a)
    a6a  = destroy(n6a)
    a1   = destroy(n1)
    a9a  = destroy(n9a)

    # make position operators
    q10a = np.sqrt(0.5)*(a10a.dag() + a10a)
    q6a  = np.sqrt(0.5)*(a6a.dag() + a6a)
    q1   = np.sqrt(0.5)*(a1.dag() + a1)
    q9a  = np.sqrt(0.5)*(a9a.dag() + a9a)
    
    # make single mode hamiltonians
    # 10a
    h10a_1 = w10a*(a10a.dag()*a10a + 0.5) + k10a_1*q10a
    h10a_2 = w10a*(a10a.dag()*a10a + 0.5) + k10a_2*q10a
    # 6a
    h6a_1 = w6a*(a6a.dag()*a6a + 0.5) + k6a_1*q6a
    h6a_2 = w6a*(a6a.dag()*a6a + 0.5) + k6a_2*q6a
    # 1
    h1_1 = w1*(a1.dag()*a1 + 0.5) + k1_1*q1
    h1_2 = w1*(a1.dag()*a1 + 0.5) + k1_2*q1
    # 9a
    h9a_1 = w9a*(a9a.dag()*a9a + 0.5) + k9a_1*q9a
    h9a_2 = w9a*(a9a.dag()*a9a + 0.5) + k9a_2*q9a
   
    # make full hamiltonian
    print('making full hamiltonian')
    # energy shift
    print('diag energy')
    p1 = tensor(phi(0),eye(n10a),eye(n6a),eye(n1),eye(n9a))
    p2 = tensor(phi(1),eye(n10a),eye(n6a),eye(n1),eye(n9a))
    H = -delta*p1
    H += delta*p2
    print('mode 10a')
    H += tensor(phi(0), h10a_1   , eye(n6a), eye(n1), eye(n9a))
    H += tensor(phi(1), h10a_2   , eye(n6a), eye(n1), eye(n9a))
    print('mode 6a')
    H += tensor(phi(0), eye(n10a), h6a_1   , eye(n1), eye(n9a))
    H += tensor(phi(1), eye(n10a), h6a_2   , eye(n1), eye(n9a))
    print('mode 1')
    H += tensor(phi(0), eye(n10a), eye(n6a), h1_1   , eye(n9a))
    H += tensor(phi(1), eye(n10a), eye(n6a), h1_2   , eye(n9a))
    print('mode 9a')
    H += tensor(phi(0), eye(n10a), eye(n6a), eye(n1), h9a_1)
    H += tensor(phi(1), eye(n10a), eye(n6a), eye(n1), h9a_1)
    print('Peierls coupling')
    H += lamda*tensor(phi(0,1), q10a, eye(n6a), eye(n1), eye(n9a))
    H += lamda*tensor(phi(1,0), q10a, eye(n6a), eye(n1), eye(n9a))
    print('Effective hamiltonian part')
    H -= gam*0.5j*tensor(eye(2), q10a*q10a, eye(n6a), eye(n1), eye(n9a))
    H -= gam*0.5j*tensor(eye(2), eye(n10a),  q6a*q6a, eye(n1), eye(n9a))
    H -= gam*0.5j*tensor(eye(2), eye(n10a), eye(n6a),   q1*q1, eye(n9a))
    H -= gam*0.5j*tensor(eye(2), eye(n10a), eye(n6a), eye(n1),  q9a*q9a)

    print('making initial wavefunction')
    # initial condition
    # e
    psie = basis(2,1)
    # 10a
    psi10a = basis(n10a,0)
    # 6a
    psi6a = basis(n6a,0)
    # 10a
    psi1 = basis(n1,0)
    # 10a
    psi9a = basis(n9a,0)
    # full psi
    psi = tensor(psie,psi10a,psi6a,psi1,psi9a)

    return H,psi,p1,p2

def main():

    # parameters
    dt = 0.50
    times = np.arange(0.0,120.0,dt)*fs2au

    # construct system
    H,psi0,p1,p2 = construct_sys()

    # run dynamics
    results = sesolve(H, psi0, times, [p1,p2], options=Options(normalize_output=False), progress_bar=True)
    output = np.zeros((len(times),3))
    output[:,0] = times/fs2au
    output[:,1] = results.expect[0]
    output[:,2] = results.expect[1]
    np.savetxt('pyr4_nh_qutip.txt', output)

if __name__ == "__main__":
    main()
