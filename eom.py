import sys
import numpy as np
from wavefunction import Wavefunction
from optools import compute_el_mes,precompute_ops,matel
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,act_density,act_projector,
                        compute_projector)
from meanfield import compute_meanfield_corr,compute_meanfield_uncorr

def eom(t,dt,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,spfs):#,cmfflag=0):
    """Evaluates the equation of motion for the coefficients and orbitals.
    """
    # precompute action of hamiltonian terms on spfs and inner products
    opspfs, opips = precompute_ops(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                   ham.ops,pbfs,spfs)
    # precompute wavefunction overlap matrices
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)
    # evaluate equation of motion for coefficients
    dA = eom_coeffs(nel,nmodes,nspfs,npbfs,ham,opips,spfovs,A)
    # evaluate equation of motion for orbitals
    dspfs = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,opspfs,opips,
                     spfovs,A,spfs)#,cmfflag)
    return dA,dspfs 

def eom_coeffs(nel,nmodes,nspfs,npbfs,ham,opips,spfovs,A):#,cmfflag):
    """Evaluates the equation of motion for the mctdh coefficients.
    """
    return -1.j*matel(nel,nmodes,nspfs,npbfs,ham.hterms,opips,spfovs,A)

def eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,opspfs,opips,spfovs,A,
             spfs):#,cmfflag):
    """Evaluates the equation of motion for the mctdh single particle funcitons.
    """
    spfsout = np.zeros(nel, dtype=np.ndarray)
    for alpha in range(nel):
        spfsout[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
        for beta in range(nel):
            spfsout[alpha] += compute_meanfield_corr(nmodes,nspfs,npbfs,spfstart,
                              spfend,alpha,beta,ham.hcterms,opspfs,opips,spfovs,
                              A,spfs)
        for mode in range(nmodes):
            nspf = nspfs[alpha,mode]
            npbf = npbfs[mode]
            ind0 = spfstart[alpha,mode]
            indf = spfend[alpha,mode]
            # compute and invert density matrix
            rho = compute_density_matrix(nspf,alpha,mode,A[alpha])
            rho = invert_density_matrix(rho)
            # act the inverted density matrix on the spfs of this mode and state
            spftmp = act_density(nspf,npbf,rho,spfsout[alpha][ind0:indf])
            # add uncorrelated terms # TODO double check these two
            #spftmp += compute_meanfield_uncorr(alpha,mode,ham.huterms,opspfs,
            #                                   spfsout[alpha][ind0:indf])
            spftmp += compute_meanfield_uncorr(alpha,mode,ham.huterms,opspfs,
                                               spfs[alpha][ind0:indf])
            # compute projector
            proj = compute_projector(nspf,npbf,spfs[alpha][ind0:indf])
            # act (1-proj) on spfs
            act_projector(nspf,npbf,proj,spftmp,spfsout[alpha][ind0:indf])
    return -1.j*spfsout

if __name__ == "__main__":

    import numpy as np
    import units
    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from pbasis import PBasis
    from time import time

    nel    = 2
    nmodes = 4
    nspfs = np.array([[7, 12, 6, 5],
                     [7, 12, 6, 5]], dtype=int)
    npbfs = np.array([22, 32, 21, 12], dtype=int)

    pbfs = list()
    pbfs.append( PBasis(['ho', 22, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 32, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 21, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 12, 1.0, 1.0]) )

    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    wf.generate_ic(1)

    w10a  = 0.09357
    w6a   = 0.0740
    w1    = 0.1273
    w9a   = 0.1568
    delta = 0.46165
    lamda = 0.1825
    k6a1  =-0.0964
    k6a2  = 0.1194
    k11   = 0.0470
    k12   = 0.2012
    k9a1  = 0.1594
    k9a2  = 0.0484

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

    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    opspfs,opips = precompute_ops(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfend,ham.ops,pbfs,wf.spfs)
    print('uncorr terms')
    print(ham.huterms)
    print('corr terms')
    print(ham.hcterms)
    for i in range(nel):
        for j in range(nmodes):
            print(i,j)
            print(opspfs[i][j].keys())
    #print(opspfs)
#    spfovs = overlap_matrices(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfs)
#    Aout = eom_coeffs(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,ham,opips,spfovs,wf.A)
#    energy = 0.0j
#    for i in range(wf.nel):
#        energy += (1.j*np.sum(wf.A[i].conj()*Aout[i])).real*units.convert_from('ev')
#    print(energy)
#
#    wf.A,wf.spfs = eom(0.0,0.5,wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfend,ham,pbfs,wf.A,wf.spfs)
#
#    print('first')
#    dwfn = eom(0.0,0.1,wf,ham)
#    print('second')
#    dwfn = eom(0.0,0.1,wf,ham)
#    eom_spfs(wf,dwfn.spfs,ham,opspfs,opips)
#    print(dwfn.spfs[0][1][0])
#    for alpha in range(nel):
#        for mode in range(nmodes):
#            print('alpha %d mode %d'%(alpha,mode))
#            for i in range(dwfn.nspf[mode]):
#                for j in range(i,dwfn.nspf[mode]):
#                    dspf = dwfn.spfs[alpha][mode][0][i*dwfn.npbf[mode]:(i+1)*dwfn.npbf[mode]]
#                    spf = wf.spfs[alpha][mode][0][i*wf.npbf[mode]:(i+1)*wf.npbf[mode]]
#                    print(i,j,np.dot(dspf.conj(),spf))
#
#    print('')
#    print('')
#    print('')
#    for alpha in range(nel):
#        for mode in range(nmodes):
#            print('alpha %d mode %d'%(alpha,mode))
#            print(dwfn.spfs[alpha][mode][0])
#            print('')

