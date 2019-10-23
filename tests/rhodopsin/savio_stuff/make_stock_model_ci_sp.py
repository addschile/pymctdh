import numpy as np
from qutip import *
import sys

cm2au = 4.556e-6 # H/cm^-1
ev2au = 0.0367493 # H/eV
fs2au = 41.3413745758 # au/fs

def construct_ops(path):

    # coupling mode parameters
    sys.stdout.write('Making coupling mode hamiltonian\n')
    sys.stdout.flush()
    nc = 24
    omega  = 0.19#*ev2au
    kappa0 = 0.0
    kappa1 = 0.0*0.095#*ev2au
    a = destroy(nc)
    qc = np.sqrt(0.5)*(a+a.dag())
    hc = omega*(a.dag()*a + 0.5)
    hc0 = hc + kappa0*qc
    hc1 = hc + kappa1*qc
    #wc0,vc0 = hc0.eigenstates()
    #wc1,vc1 = hc1.eigenstates()
    #hc0e = Qobj(np.diag(wc0))
    #hc1e = Qobj(np.diag(wc1))

    # rotor mode parameters
    sys.stdout.write('Making rotor mode hamiltonian\n')
    sys.stdout.flush()
    nphi  = (int(2*150))+1
    nm    = int((nphi-1)/2)
    minv  = 1.43e-3#*ev2au
    E0    = 0.0
    E1    = 2.00#*ev2au
    W0    = 2.3#*ev2au
    W1    = 1.50#*ev2au
    lamda = 0.19#*ev2au

    nsite = 2*nphi*nc

    # hphi
    cosphi = np.zeros((nphi,nphi),dtype=complex)
    for i in range(nphi-1):
        cosphi[i,i+1] = 0.5
        cosphi[i+1,i] = 0.5
    cosphi = Qobj(cosphi)

    tphi = np.zeros((nphi,nphi),dtype=complex)
    for i in range(-nm,nm+1):
        tphi[i+nm,i+nm] = -float(i)**2.
    tphi = Qobj(tphi)

    ident = qeye(nphi)
    qphi = ident-cosphi
    V0 = E0*ident + 0.5*W0*(ident-cosphi)
    V1 = E1*ident - 0.5*W1*(ident-cosphi)
    hphi0 = -0.5*minv*tphi + V0
    hphi1 = -0.5*minv*tphi + V1
    wphi0,vphi0 = hphi0.eigenstates()
    wphi1,vphi1 = hphi1.eigenstates()
    hphi0e = Qobj(np.diag(wphi0))
    hphi1e = Qobj(np.diag(wphi1))

    # FC overlap matrix
    sys.stdout.write('Making overlap matrices\n')
    sys.stdout.flush()
    overlaps = np.zeros((nphi,nphi),dtype=complex)
    for i in range(nphi):
        for j in range(nphi):
            overlaps[i,j] = (vphi0[i].dag()*vphi1[j])[0,0]
    overlaps = Qobj(overlaps)
    #qcoverlaps = np.zeros((nc,nc),dtype=complex)
    #coverlaps = np.zeros((nc,nc),dtype=complex)
    #for i in range(nc):
    #    for j in range(nc):
    #        coverlaps[i,j] = (vc0[i].dag()*vc1[j])[0,0]
    #        qcoverlaps[i,j] = (vc0[i].dag()*qc*vc1[j])[0,0] 
    #        ## conical intersection 
    #        #for k in range(nc):
    #        #    for l in range(nc):
    #        #        qcoverlaps[i,j] += (qc[k,l]*np.conj(vc0[k,i])*vc1[l,j])
    #coverlaps = Qobj(coverlaps)
    #qcoverlaps = Qobj(qcoverlaps)

    # Hamiltonian in full space
    sys.stdout.write('Making full hamiltonian\n')
    sys.stdout.flush()
    # relevant stuff
    eye_e   = qeye(2)
    eye_c   = qeye(nc)
    eye_phi = qeye(nphi)
    phi0    = fock_dm(2,0)
    phi1    = fock_dm(2,1)
    phi01   = basis(2,0)*basis(2,1).dag()
    phi10   = basis(2,1)*basis(2,0).dag()
    # diabatic hamiltonian stuff
    H  = (E0+0.5*W0)*tensor(phi0, eye_phi, eye_c)
    H += (E1-0.5*W1)*tensor(phi1, eye_phi, eye_c)
    H  = tensor(phi0, hphi0e, eye_c)
    H += tensor(phi1, hphi1e, eye_c)
    H += tensor(phi0, eye_phi, hc0)
    H += tensor(phi1, eye_phi, hc1)
    # electronic coupling
    H += lamda*tensor(phi01, overlaps, qc)
    H += lamda*tensor(phi10, overlaps.dag(), qc)

    ### initial condition
    sys.stdout.write('Creating initial condition\n')
    sys.stdout.flush()
    # cis excitation
    psie = np.zeros((2,1),dtype=complex)
    psie[1,0] = 1.
    psie = Qobj(psie)
    psiphi = np.zeros((nphi,1),dtype=complex)
    for j in range(nphi):
        psiphi[j,0] += np.conj(overlaps[0,j])
    psiphi = Qobj(psiphi)
    psic = np.zeros((nc,1),dtype=complex)
    #for j in range(nc):
    #    psic[j,0] += np.conj(coverlaps[0,j])
    psic[0,0] = 1.
    psic = Qobj(psic)
    psi_0 = tensor(psie,psiphi,psic)

    print(psi_0.dag()*H*psi_0)

    return

if __name__ == '__main__':
    path = ''
    construct_ops(path)
