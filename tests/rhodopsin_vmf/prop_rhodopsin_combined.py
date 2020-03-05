import numpy as np
import sys
sys.path.append('/Users/addisonschile/Software/ode_pymctdh')
import units
from wavefunction import Wavefunction
from hamiltonian import Hamiltonian
from pbasis import PBasis
from vmfpropagate import vmfpropagate

if __name__ == "__main__":

    nel    = 2
    nmodes = 1
    nspfs = np.array([[10],
                      [10]], dtype=int)
    npbfs = [[24, 151]]

    # TODO need to be able to combine modes with different pbfs
    pbfs = list()
    pbfs.append( PBasis(['ho',          24, 1.0, 1.0]) )
    pbfs.append( PBasis(['plane wave', 151, 1.0],sparse=True) )

    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    wf.generate_ic(1)

    # reshpae y into A tensor and spfs
    A = np.zeros(2, dtype=np.ndarray)
    spfs = np.zeros(2, dtype=np.ndarray)
    for alpha in range(wf.nel):
        shaper = ()
        for mode in range(wf.nmodes):
            shaper += (wf.nspfs[alpha,mode],)
        # set A
        ind0 = wf.psistart[0,alpha]
        indf = wf.psiend[0,alpha]
        A[alpha] = np.reshape(wf.psi[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = wf.psistart[1,alpha]
        indf = wf.psiend[1,alpha]
        spfs[alpha] = wf.psi[ind0:indf]

    psiphi = np.load('ground_state_initial_condition.npy')
    # reset the relevant spf
    ind0 = wf.spfstart[1,1]
    npbf = wf.npbfs[1]
    nrm = np.dot(psiphi[75:-75,0].conj(),psiphi[75:-75,0])
    spfs[1][ind0:ind0+npbf] = psiphi[75:-75,0]/np.sqrt(nrm)
    #nrm = np.dot(psiphi[50:-50,0].conj(),psiphi[50:-50,0])
    #spfs[1][ind0:ind0+npbf] = psiphi[50:-50,0]/np.sqrt(nrm)

    # reshape everything for output
    for alpha in range(nel):
        ind0 = wf.psistart[0,alpha]
        indf = wf.psiend[0,alpha]
        wf.psi[ind0:indf] = A[alpha].ravel()
        ind0 = wf.psistart[1,alpha]
        indf = wf.psiend[1,alpha]
        wf.psi[ind0:indf] = spfs[alpha]

    wf.orthonormalize_spfs()

    wc     = 0.19
    kappa0 = 0.0
    kappa1 = 0.095
    minv   = 1.43e-3
    E0     = 0.0
    E1     = 2.00
    W0     = 2.3
    W1     = 1.50
    lamda  = 0.19

    hterms = []
    # electronic energy shifts
    hterms.append({'coeff': E0+0.5*W0, 'units': 'ev', 'elop': '0,0'})
    hterms.append({'coeff': E1-0.5*W1, 'units': 'ev', 'elop': '1,1'})
    ## coupling mode harmonic oscillator potential
    hterms.append({'coeff':     wc, 'units': 'ev', 'modes': 0, 'ops':  'KE'})
    hterms.append({'coeff': 0.5*wc, 'units': 'ev', 'modes': 0, 'ops': 'q^2'})
    hterms.append({'coeff': kappa1, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q'})
    hterms.append({'coeff':  lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': 'q'})
    # torsional mode plane wave potential 
    hterms.append({'coeff': -0.5*minv, 'units': 'ev', 'modes': 1, 'ops': 'KE'})
    hterms.append({'coeff':   -0.5*W0, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': 'cos'})
    hterms.append({'coeff':    0.5*W1, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': 'cos'})

    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    dt = 1.0
    times = np.arange(0.0,4000.,dt)*units.convert_to('fs')

    #wf = vmfpropagate(times, ham, pbfs, wf, 'rhodopsin_diabatic_pops_test.txt')
    #wf = vmfpropagate(times, ham, pbfs, wf, 'rhodopsin_diabatic_pops.txt')
    #wf = vmfpropagate(times, ham, pbfs, wf, 'rhodopsin_diabatic_pops_small.txt')
    wf = vmfpropagate(times, ham, pbfs, wf, 'rhodopsin_diabatic_pops_really_small.txt')
