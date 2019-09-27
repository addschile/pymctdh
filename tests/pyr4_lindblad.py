import numpy as np
from qutip import *
import sys
sys.path.append('/Users/addisonschile/Software/')
import qdynos.constants as const

def construct_sys():

    # dimensions
    n10a = 25
    n6a  = 35
    n1   = 25
    n9a  = 20
    nstates = 2*n10a*n6a*n1*n9a
    # energies
    delta = 0.46165*const.ev2au
    # frequencies
    w10a = 0.09357*const.ev2au
    w6a  = 0.0740*const.ev2au
    w1   = 0.1273*const.ev2au
    w9a  = 0.1568*const.ev2au
    # holstein couplings
    # H_11
    k10a_1 = 0.0*const.ev2au
    k6a_1  = -0.0964*const.ev2au
    k1_1   = 0.0470*const.ev2au
    k9a_1  = 0.1594*const.ev2au
    # H_22
    k10a_2 = 0.0*const.ev2au
    k6a_2  = 0.1194*const.ev2au
    k1_2   = 0.2012*const.ev2au
    k9a_2  = 0.0484*const.ev2au
    # peierls coupling
    lamda = 0.1825*const.ev2au
    
    # make position operators
    a10a = destroy(n10a)
    q10a = np.sqrt(0.5)*(a10a + a10a.dag())
    a6a = destroy(n6a)
    q6a = np.sqrt(0.5)*(a6a + a6a.dag())
    a1 = destroy(n1)
    q1 = np.sqrt(0.5)*(a1 + a1.dag())
    a9a = destroy(n9a)
    q9a = np.sqrt(0.5)*(a9a + a9a.dag())

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
    p1 = tensor(fock_dm(2,0),qeye(n10a),qeye(n6a),qeye(n1),qeye(n9a))
    p2 = tensor(fock_dm(2,1),qeye(n10a),qeye(n6a),qeye(n1),qeye(n9a))
    H = -delta*p1
    H += delta*p2
    print('mode 10a')
    H += tensor(fock_dm(2,0), h10a_1   , qeye(n6a), qeye(n1), qeye(n9a))
    H += tensor(fock_dm(2,1), h10a_2   , qeye(n6a), qeye(n1), qeye(n9a))
    print('mode 6a')
    H += tensor(fock_dm(2,0), qeye(n10a), h6a_1   , qeye(n1), qeye(n9a))
    H += tensor(fock_dm(2,1), qeye(n10a), h6a_2   , qeye(n1), qeye(n9a))
    print('mode 1')
    H += tensor(fock_dm(2,0), qeye(n10a), qeye(n6a), h1_1   , qeye(n9a))
    H += tensor(fock_dm(2,1), qeye(n10a), qeye(n6a), h1_2   , qeye(n9a))
    print('mode 9a')
    H += tensor(fock_dm(2,0), qeye(n10a), qeye(n6a), qeye(n1), h9a_1)
    H += tensor(fock_dm(2,1), qeye(n10a), qeye(n6a), qeye(n1), h9a_1)
    print('Peierls coupling')
    H += lamda*tensor(basis(2,0)*basis(2,1).dag(), q10a, qeye(n6a), qeye(n1), qeye(n9a))
    H += lamda*tensor(basis(2,1)*basis(2,0).dag(), q10a, qeye(n6a), qeye(n1), qeye(n9a))
    #print('Effective hamiltonian part')
    #H -= 0.1*const.ev2au*0.5j*kron(eye(2),q10a.dot(q10a),     eye(n6a),    eye(n1),     eye(n9a))
    #H -= 0.1*const.ev2au*0.5j*kron(eye(2),     eye(n10a), q6a.dot(q6a),    eye(n1),     eye(n9a))
    #H -= 0.1*const.ev2au*0.5j*kron(eye(2),     eye(n10a),     eye(n6a), q1.dot(q1),     eye(n9a))
    #H -= 0.1*const.ev2au*0.5j*kron(eye(2),     eye(n10a),     eye(n6a),    eye(n1), q9a.dot(q9a))

    print('making qs')
    q10a = tensor(qeye(2),      q10a, qeye(n6a), qeye(n1), qeye(n9a))
    q6a  = tensor(qeye(2), qeye(n10a),      q6a, qeye(n1), qeye(n9a))
    q1   = tensor(qeye(2), qeye(n10a), qeye(n6a),      q1, qeye(n9a))
    q9a  = tensor(qeye(2), qeye(n10a), qeye(n6a), qeye(n1),      q9a)

    print('making initial wavefunction')
    # initial condition
    # e
    psie = basis(2,0)
    # 10a
    psi10a = basis(n10a,0)
    # 6a
    psi6a = basis(n6a,0)
    # 10a
    psi1 = basis(n1,0)
    # 10a
    psi9a = basis(n9a,0)
    # full psi
    rho = tensor(psie,psi10a,psi6a,psi1,psi9a)
    #rho = rho*rho.dag()

    return H,rho,p1,p2,q10a,q6a,q1,q9a

def main():

    # parameters
    dt = 0.5
    times = np.arange(0.0,120.0,dt)*const.fs2au

    # construct system
    H,rho0,p1,p2,q10a,q6a,q1,q9a = construct_sys()

    # scale by 
    q10a *= np.sqrt(0.1)
    q6a *= np.sqrt(0.1)
    q1 *= np.sqrt(0.1)
    q9a *= np.sqrt(0.1)

    print('running dynamics')
    result = mcsolve(H, rho0, times, [q10a,q6a,q1,q9a], [p1,p2], ntraj=1, progress_bar=True)
    np.savetxt(results.expect[0], 'pyr4_lindblad_p1.txt')
    np.savetxt(results.expect[1], 'pyr4_lindblad_p2.txt')

    #result = mesolve(H, rho0, times, [q10a,q6a,q1,q9a], [p1,p2])#, progress_bar=True)
    #np.savetxt(results.expect[0], 'pyr4_lindblad_p1.txt')
    #np.savetxt(results.expect[1], 'pyr4_lindblad_p2.txt')

if __name__ == "__main__":
    main()
