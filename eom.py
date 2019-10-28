import sys
import numpy as np
from wavefunction import Wavefunction
from optools import compute_el_mes,precompute_ops,matel
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,act_density,act_projector,
                        compute_projector,project)
from meanfield import (compute_meanfield_corr,compute_meanfield_elcorr,
                       compute_meanfield_uncorr,compute_meanfield_eluncorr,
                       act_meanfield)
                       
def vmfeom(t, y, npsi, nel, nmodes, nspfs, npbfs, psistart, psiend, spfstart, spfend,
           ham, pbfs):
    """
    """

    # reshpae y into A tensor and spfs
    A = np.zeros(2, dtype=np.ndarray)
    spfs = np.zeros(2, dtype=np.ndarray)
    for alpha in range(nel):
        shaper = ()
        for mode in range(nmodes):
            shaper += (nspfs[alpha,mode],)
        # set A
        ind0 = psistart[0,alpha]
        indf = psiend[0,alpha]
        A[alpha] = np.reshape(y[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = psistart[1,alpha]
        indf = psiend[1,alpha]
        spfs[alpha] = y[ind0:indf]

    # using vmf equation of motion
    dA,dspfs = fulleom(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,spfs)

    # reshape everything for output
    dy = np.zeros(npsi, dtype=complex)
    for alpha in range(nel):
        ind0 = psistart[0,alpha]
        indf = psiend[0,alpha]
        dy[ind0:indf] = dA[alpha].ravel()
        ind0 = psistart[1,alpha]
        indf = psiend[1,alpha]
        dy[ind0:indf] = dspfs[alpha]

    return dy

def fulleom(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,spfs):
    """Evaluates the equation of motion for the coefficients and orbitals.
    """
    # precompute stuff needed for equations of motion
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                       spfstart,spfend,ham.huterms,ham.hcterms,
                                       pbfs,spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)
    # evaluate equations of motion for coefficients
    dA = eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                    ham.hcelterms,spfovs,A)
    # evaluate equations of motion for spfs
    dspfs = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,copips,
                     ham.huelterms,ham.hcelterms,spfovs,A,spfs)
    #np.set_printoptions(threshold=np.inf)
    #print(dspfs)
    ##print(dA)
    #raise ValueError
    return -1.j*dA,-1.j*dspfs 

def cmfeom_coeffs(t,A,npsi,nel,nmodes,nspfs,npbfs,psistart,psiend,huelterms,
                  hcelterms,uopips,copips,spfovs):
    """Evaluates the equation of motion for the mctdh coefficients.
    """
    #A = np.zeros(2, dtype=np.ndarray)
    #for alpha in range(nel):
    #    shaper = ()
    #    for mode in range(nmodes):
    #        shaper += (nspfs[alpha,mode],)
    #    ind0 = psistart[0,alpha]
    #    indf = psiend[0,alpha]
    #    A[alpha] = np.reshape(y[ind0:indf], shaper, order='C')

    # using equation of motion for coefficients
    dA = eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,spfovs,A)
    return dA

    ## reshape everything for output
    #dy = np.zeros(npsi, dtype=complex)
    #for alpha in range(nel):
    #    ind0 = psistart[0,alpha]
    #    indf = psiend[0,alpha]
    #    dy[ind0:indf] = dA[alpha].ravel()

    #return dy

def eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,spfovs,A):
    """Evaluates the equation of motion for the mctdh coefficients.
    """
    return matel(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,
                      spfovs,A)

#def cmfeom_spfs(t,y,npsi,nel,nmodes,nspfs,npbfs,spfstart,spfend,huelterms,
#                hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A,mfs,rhos,projs):
def cmfeom_spfs(t,y,npsi,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,mfs,rhos):
    """Evaluates the equation of motion for the mctdh coefficients.
    """
    # reshpae y into spfs
    spfs = np.zeros(2, dtype=np.ndarray)
    for alpha in range(nel):
        # set spfs
        ind0 = spfstart[alpha,0]
        indf = spfend[alpha,-1]
        if alpha!=0:
            ind0 += spfend[alpha-1,-1]
            indf += spfend[alpha-1,-1]
        spfs[alpha] = y[ind0:indf]

    # precompute stuff needed for equations of motion
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                       spfstart,spfend,ham.huterms,ham.hcterms,
                                       pbfs,spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)

    # using equation of motion for spfs
    #dspfs = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,
    #                 copips,huelterms,hcelterms,spfovs,A,spfs,mfs=mfs,rhos=rhos,
    #                 projs=projs)
    dspfs = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,
                     copips,ham.huelterms,ham.hcelterms,spfovs,A,spfs,mfs=mfs,
                     rhos=rhos)

    #np.set_printoptions(threshold=np.inf)
    #print(dspfs)
    #raise ValueError

    # reshape everything for output
    dy = np.zeros(npsi, dtype=complex)
    for alpha in range(nel):
        ind0 = spfstart[alpha,0]
        indf = spfend[alpha,-1]
        dy[ind0:indf] = dspfs[alpha]

    return -1.j*dy

# TODO figure out efficient way to take out A dependence
def eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,copips,
             huelterms,hcelterms,spfovs,A,spfs,mfs=None,rhos=None,projs=None):
    """Evaluates the equation of motion for the mctdh single particle funcitons.
    """
    # create output array
    spfsout = np.zeros(nel, dtype=np.ndarray)
    for alpha in range(nel):
        spfsout[alpha] = np.zeros_like(spfs[alpha], dtype=complex)

    # act the correlated terms
    if not copips is None:
        if mfs is None:
            compute_meanfield_corr(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                   copspfs,copips,spfovs,A,spfs,spfsout)
        else:
            act_meanfield(nel,nmodes,nspfs,npbfs,spfstart,spfend,mfs,copspfs,
                          copips,spfs,spfsout)

    # act the correlated electronic terms
    if len(hcelterms) != 0:
        compute_meanfield_elcorr(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                 hcelterms,spfovs,A,spfs,spfsout)

    # compute, invert, and act density matrices
    for alpha in range(nel):
        for mode in range(nmodes):
            nspf = nspfs[alpha,mode]
            npbf = npbfs[mode]
            ind0 = spfstart[alpha,mode]
            indf = spfend[alpha,mode]
            if rhos is None: 
                # compute and invert density matrix
                rho = compute_density_matrix(nspf,alpha,nmodes,mode,A[alpha])
                rho = invert_density_matrix(rho)
                # act the inverted density matrix on the spfs of this mode and state
                spfsout[alpha][ind0:indf] = act_density(nspf,npbf,rho,
                                               spfsout[alpha][ind0:indf])
            else:
                # act the inverted density matrix on the spfs of this mode and state
                spfsout[alpha][ind0:indf] = act_density(nspf,npbf,rhos[alpha][mode],
                                               spfsout[alpha][ind0:indf])

    # add uncorrelated terms
    if not uopspfs is None:
        compute_meanfield_uncorr(nel,nmodes,spfstart,spfend,uopspfs,spfs,spfsout)

    # add uncorrelated electronic terms
    if len(huelterms) != 0:
        compute_meanfield_eluncorr(nel,nmodes,spfstart,spfend,huelterms,spfs,spfsout)

    # compute and act projectors
    for alpha in range(nel):
        for mode in range(nmodes):
            nspf = nspfs[alpha,mode]
            npbf = npbfs[mode]
            ind0 = spfstart[alpha,mode]
            indf = spfend[alpha,mode]
            ## compute projector
            #proj = compute_projector(nspf,npbf,spfs[alpha][ind0:indf])
            ## act (1-proj) on spfs
            #act_projector(nspf,npbf,proj,spfsout[alpha][ind0:indf])
            project(nspf,npbf,spfs[alpha][ind0:indf],spfsout[alpha][ind0:indf])
            #if projs is None:
            #    # compute action of projector
            #    project(nspf,npbf,spfs[alpha][ind0:indf],spfsout[alpha][ind0:indf])
            ##    # compute projector
            ##    #proj = compute_projector(nspf,npbf,spfs[alpha][ind0:indf])
            ##    # act (1-proj) on spfs
            ##    #act_projector(nspf,npbf,proj,spfsout[alpha][ind0:indf])
            ##    project(nspf,npbf,spfs[alpha][ind0:indf],spfsout[alpha][ind0:indf])
            #else:
            #    # act (1-proj) on spfs
            #    act_projector(nspf,npbf,projs[alpha][mode],spfsout[alpha][ind0:indf])

    return spfsout

if __name__ == "__main__":

    from time import time
    import numpy as np
    import units
    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from pbasis import PBasis
    from time import time

    nel    = 2
    nmodes = 4
    nspfs = np.array([[8, 13, 7, 6],
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

    # evaluate eom coeffs and compute energy
    uopspfs,copspfs,uopips,copips = precompute_ops(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,
                                        wf.spfstart,wf.spfend,ham.huterms,ham.hcterms,pbfs,wf.spfs)
    spfovs = overlap_matrices(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfs)
    dA = eom_coeffs(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,uopips,copips,ham.huelterms,ham.hcelterms,spfovs,wf.A)
    energy = 0.0j
    for i in range(wf.nel):
        energy += (1.j*np.sum(wf.A[i].conj()*dA[i])).real*units.convert_from('ev')
    print(energy)

    dspfs = eom_spfs(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfend,uopspfs,copspfs,copips,
                     ham.huelterms,ham.hcelterms,spfovs,wf.A,wf.spfs)
    for alpha in range(nel):
        print('alpha %d'%(alpha))
        for mode in range(nmodes):
            print('mode %d'%(mode))
            ind = wf.spfstart[alpha,mode]
            npbf = wf.npbfs[mode]
            nspf = wf.nspfs[alpha,mode]
            ind = wf.spfstart[alpha,mode]
            for n in range(nspf):
                dspf = dspfs[alpha][ind:ind+npbf]
                ind += npbf
                x = np.nonzero(dspf)
                print(dspf[x])
                #print(dspf)
