import numpy as np
import sys

cm2au = 4.556e-6 # H/cm^-1
ev2au = 0.0367493 # H/eV
fs2au = 41.3413745758 # au/fs

def construct_ops(path):

    # coupling mode parameters
    sys.stdout.write('Making coupling mode hamiltonian\n')
    sys.stdout.flush()
    nc = 24
    omega  = 0.19*ev2au
    kappa0 = 0.0
    kappa1 = 0.095*ev2au
    qc = np.zeros((nc,nc),dtype=complex)
    for i in range(nc-1):
        qc[i,i+1] = np.sqrt(float(i+1)/2.)
        qc[i+1,i] = np.sqrt(float(i+1)/2.)
    hc = np.zeros((nc,nc),dtype=complex)
    for i in range(nc):
        hc[i,i] = omega*(float(i)+0.5)
    hc0 = hc + kappa0*qc
    hc1 = hc + kappa1*qc
    wc0,vc0 = np.linalg.eigh(hc0)
    wc1,vc1 = np.linalg.eigh(hc1)

    # rotor mode parameters
    sys.stdout.write('Making rotor mode hamiltonian\n')
    sys.stdout.flush()
    nphi  = (int(2*150))+1
    nm    = int((nphi-1)/2)
    minv  = 1.43e-3*ev2au
    E0    = 0.0
    E1    = 2.00*ev2au
    W0    = 2.3*ev2au
    W1    = 1.50*ev2au
    lamda = 0.19*ev2au

    nsite = 2*nphi*nc

    # hphi
    cosphi = np.zeros((nphi,nphi),dtype=complex)
    for i in range(nphi-1):
        cosphi[i,i+1] = 0.5
        cosphi[i+1,i] = 0.5

    tphi = np.zeros((nphi,nphi),dtype=complex)
    for i in range(-nm,nm+1):
        tphi[i+nm,i+nm] = -float(i)**2.

    ident = np.identity(nphi,dtype=complex)
    qphi = ident-cosphi
    V0 = E0*ident + 0.5*W0*(ident-cosphi)
    V1 = E1*ident - 0.5*W1*(ident-cosphi)
    hphi0 = -0.5*minv*tphi + V0
    hphi1 = -0.5*minv*tphi + V1
    wphi0,vphi0 = np.linalg.eigh(hphi0)
    wphi1,vphi1 = np.linalg.eigh(hphi1)

    # FC overlap matrix
    overlaps = np.zeros((nphi,nphi),dtype=complex)
    for i in range(nphi):
        for j in range(nphi):
            overlaps[i,j] = np.sum((np.conj(vphi0[:,i])*vphi1[:,j])[:])
    psiphi = np.zeros((nphi,1),dtype=complex)
    for j in range(nphi):
        psiphi[j,0] += np.conj(overlaps[0,j])
    psiphi = np.dot(vphi1, psiphi)
    np.save(path+'ground_state_initial_condition.npy',psiphi)

    return

if __name__ == '__main__':
    path = ''
    construct_ops(path)
