import sys
import numpy as np
from .wavefunction import Wavefunction
from .optools import compute_el_mes,precompute_ops,compute_uopspfs,matel
from .cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,act_density,act_projector,
                        compute_projector,project,reshape_wf,reshape_wf_back)
from .meanfield import (compute_meanfield_corr,compute_meanfield_elcorr,
                       compute_meanfield_uncorr,compute_meanfield_eluncorr,
                       act_meanfield)
                       
def vmfeom(t, y, npsi, nel, nmodes, nspfs, npbfs, psistart, psiend, spfstart, spfend,
           ham, pbfs):
    """Evaluates the variational mean field equation of motion.
    """

    A,spfs = reshape_wf(nel,nmodes,nspfs,psistart,psiend,y)

    # using vmf equation of motion
    dA,dspfs = fulleom(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,spfs)

    return reshape_wf_back(nel,npsi,psistart,psiend,dA,dspfs)

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
    return -1.j*dA,-1.j*dspfs 

def cmfeom_coeffs(t,A,npsi,nel,nmodes,nspfs,npbfs,psistart,psiend,huelterms,
                  hcelterms,uopips,copips,spfovs):
    """Evaluates the equation of motion for the mctdh coefficients within the
    CMF approximation.
    """
    # using equation of motion for coefficients
    dA = eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,spfovs,A)
    return dA

def eom_coeffs(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,spfovs,A):
    """Evaluates the matrix elements for the mctdh coefficients.
    """
    return matel(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,
                      spfovs,A)

def cmfeom_spfs(t,y,npsi,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,
                copspfs,copips,spfovs,mfs,rhos):
    """Evaluates the equation of motion for the mctdh single-particle functions
    within the CMF approximation.
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
    #uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
    #                                   spfstart,spfend,ham.huterms,ham.hcterms,
    #                                   pbfs,spfs)
    #spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)

    uopspfs,copspfs,_uopips,_copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                       spfstart,spfend,ham.huterms,ham.hcterms,
                                       pbfs,spfs)
#    # only precompute opspfs for uncorrelated terms
#    uopspfs = compute_uopspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,
#                               ham.huterms,pbfs,spfs)

    # using equation of motion for spfs
    dspfs = eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,
                     copips,ham.huelterms,ham.hcelterms,spfovs,A,spfs,mfs=mfs,
                     rhos=rhos)

    np.set_printoptions(threshold=np.inf)
    #print(dspfs)
    #raise ValueError

    # reshape everything for output
    dy = np.zeros(npsi, dtype=complex)
    for alpha in range(nel):
        ind0 = spfstart[alpha,0]
        indf = spfend[alpha,-1]
        dy[ind0:indf] = dspfs[alpha]

    return -1.j*dy

#def cmfeom_spfsnew(t,y,npsi,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,mfs,rhos):
#    """Evaluates the equation of motion for the mctdh coefficients.
#    """
#    # reshpae y into spfs
#    spfs = np.zeros(2, dtype=np.ndarray)
#    for alpha in range(nel):
#        # set spfs
#        ind0 = spfstart[alpha,0]
#        indf = spfend[alpha,-1]
#        if alpha!=0:
#            ind0 += spfend[alpha-1,-1]
#            indf += spfend[alpha-1,-1]
#        spfs[alpha] = y[ind0:indf]
#
#    # create output array
#    dspfs = np.zeros(nel, dtype=np.ndarray)
#    for alpha in range(nel):
#        dspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
#
#    # uncorrelated terms
#    if not copips is None:
#        act_meanfield(nel,nmodes,nspfs,npbfs,spfstart,spfend,mfs,copspfs,
#                      copips,spfs,dspfs)
#
#    # act density matrices
#    for alpha in range(nel):
#        for mode in range(nmodes):
#            nspf = nspfs[alpha,mode]
#            npbf = npbfs[mode]
#            ind0 = spfstart[alpha,mode]
#            indf = spfend[alpha,mode]
#            # act the inverted density matrix on the spfs of this mode and state
#            dspfs[alpha][ind0:indf] = act_density(nspf,npbf,rhos[alpha][mode],
#                                           dspfs[alpha][ind0:indf])
#
#    # TODO only precompute the uncorrelated stuff
#    uopspfs = compute_uopspfs(nel,nmodes,nspfs,npbfs,
#
#    # add uncorrelated terms
#    if not uopspfs is None:
#        compute_meanfield_uncorr(nel,nmodes,spfstart,spfend,uopspfs,spfs,dspfs)
#
#    # add uncorrelated electronic terms
#    if len(huelterms) != 0:
#        compute_meanfield_eluncorr(nel,nmodes,spfstart,spfend,huelterms,spfs,dspfs)
#
#    # compute and act projectors
#    for alpha in range(nel):
#        for mode in range(nmodes):
#            nspf = nspfs[alpha,mode]
#            npbf = npbfs[mode]
#            ind0 = spfstart[alpha,mode]
#            indf = spfend[alpha,mode]
#            # compute projector and act 1-projector
#            project(nspf,npbf,spfs[alpha][ind0:indf],dspfs[alpha][ind0:indf])
#
#    # reshape everything for output
#    dy = np.zeros(npsi, dtype=complex)
#    for alpha in range(nel):
#        ind0 = spfstart[alpha,0]
#        indf = spfend[alpha,-1]
#        dy[ind0:indf] = dspfs[alpha]
#
#    return -1.j*dy

# TODO figure out efficient way to take out A dependence
def eom_spfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,copips,
             huelterms,hcelterms,spfovs,A,spfs,mfs=None,rhos=None,projs=None):
    """Evaluates the equation of motion for the mctdh single particle funcitons
    within the VMF integration scheme.
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
            # compute projector and act 1-projector
            project(nspf,npbf,spfs[alpha][ind0:indf],spfsout[alpha][ind0:indf])

    return spfsout
