import numpy as np
from functools import lru_cache
from copy import deepcopy
from .cy.wftools import spf_innerprod,overlap_matrices2,compute_projector
from .cy.tensorutils import matelcontract,atensorcontract

@lru_cache(maxsize=None,typed=False)
def isdiag(op):
    if op == '1':
        return True
    elif op == 'sz':
        return True
    elif op == 'sx':
        return False 
    elif op == 'sy':
        return False 
    elif ',' in op:
        op_ = op.split(',')
        if op_[0]==op_[1]:
            return True
        else:
            return False
    else:
        return ValueError('Invalid electronic operator type.')

@lru_cache(maxsize=None,typed=False)
def elop_state(alpha,op):
    if ',' in op:
        op_ = op.split(',')
        if int(op_[1])==alpha: return 1.0
        else: return 0.0
    else:
        return 1.0

@lru_cache(maxsize=None,typed=False)
def compute_el_mes(alpha,beta,op):
    """Function that computes the matrix element between electronic states
    for nonadiabatic mctdh dynamics.
    """
    if op == '1':
        if alpha == beta: return 1.0
        else: return 0.0
    elif op == 'sz':
        if alpha != beta: return 0.0
        elif alpha == beta:
            if alpha == 0: return 1.0
            elif alpha == 1: return -1.0
    elif op == 'sx':
        if alpha == beta: return 0.0
        else: return 1.0
    elif op == 'sy':
        if alpha == beta: return 0.0
        else:
            if alpha == 0: return -1.0j
            else: return 1.0j
    elif ',' in op:
        op_ = op.split(',')
        if int(op_[0])==alpha and int(op_[1])==beta: return 1.0
        else: return 0.0
    else:
        return ValueError('Invalid electronic operator type.')

def precompute_ops(nel,nmodes,nspfs,npbfs,spfstart,spfend,huterms,hcterms,pbfs,*spfs):
    """Precomputes actions of hamiltonian operators on the spfs and the inner
    products.
    """
    # compute actions of the operators on spfs
    #uopspfs,copspfs = compute_opspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,
    #                                 huterms,hcterms,pbfs,spfs[0])
    uopspfs = compute_uopspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,huterms,
                              pbfs,spfs[0])
    copspfs = compute_copspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,hcterms,
                              pbfs,spfs[0])
    # now compute the matrices of their inner products
    uopips,copips = compute_opips(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                  uopspfs,copspfs,hcterms,spfs[-1])
    return uopspfs , copspfs , uopips , copips

def compute_uopspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,huterms,pbfs,spfs):
    """Computes all the individual operators that act on the spfs for only the 
    uncorrelated hamiltonian terms.
    """
    # compute uncorrelated op*spfs
    #if len(huterms) == 0:
    #    uopspfs = None
    #else:
    #    uopspfs = np.zeros(nel, dtype=np.ndarray)
    #    for alpha in range(nel):
    #        uopspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
    #    # loop over all the uncorrelated terms in the hamiltonian
    #    for hterm in huterms:
    #        for alpha in range(nel):
    #            # does this term act on this electronic state?
    #            elop = hterm['elop']
    #            if elop_state(alpha,elop) != 0:
    #                mel = compute_el_mes(alpha,alpha,elop)
    #                opspfs = uopspfs[alpha]
    #                coeff = hterm['coeff']
    #                if 'modes' in hterm:
    #                    # act it on the mode
    #                    mode  = hterm['modes'][0]
    #                    nspf  = nspfs[alpha,mode]
    #                    npbf  = npbfs[mode]
    #                    ind   = spfstart[alpha,mode]
    #                    op    = hterm['ops'][0]
    #                    for n in range(nspf):
    #                        spf = spfs[alpha][ind:ind+npbf]
    #                        opspfs[ind:ind+npbf] += coeff*mel*pbfs[mode].operate(spf,op)
    #                        ind += npbf
    #                else:
    #                    # multiply coefficient to spfs
    #                    opspfs += coeff*mel*spfs[alpha]
    if huterms is None:
        uopspfs = None
    else:
        uopspfs = np.zeros(nel, dtype=np.ndarray)
        for alpha in range(nel):
            uopspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
            opspfs = uopspfs[alpha]
            for mode in range(nmodes):
                nspf = nspfs[alpha,mode]
                npbf = npbfs[mode]
                ind  = spfstart[alpha,mode]
                for n in range(nspf):
                    spf = spfs[alpha][ind:ind+npbf]
                    opspfs[ind:ind+npbf] += pbfs[mode].operate1b(spf, alpha)
                    ind += npbf
    return uopspfs

def compute_copspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,hcterms,pbfs,spfs):
    # compute correlated op*spfs
    if len(hcterms) == 0:
        copspfs = None
    else:
        copspfs = np.zeros(len(hcterms), dtype=np.ndarray)
        # loop over all the correlated terms in the hamiltonian
        for i,hterm in enumerate(hcterms):
            # loop over each electronic state
            opspfs = np.zeros(nel, dtype=np.ndarray)
            for alpha in range(nel):
                # does this term act on this electronic state?
                elop = hterm['elop']
                if elop_state(alpha,elop) != 0:
                    opspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
                    coeff = hterm['coeff']
                    # act it on the modes
                    modes = hterm['modes']
                    for j,mode in enumerate(modes):
                        nspf = nspfs[alpha,mode]
                        npbf = npbfs[mode]
                        ind  = spfstart[alpha,mode]
                        op   = hterm['ops'][j]
                        for n in range(nspf):
                            spf = spfs[alpha][ind:ind+npbf]
                            opspfs[alpha][ind:ind+npbf] += pbfs[mode].operate(spf,op)
                            ind += npbf
            copspfs[i] = opspfs
    return copspfs

def compute_opspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,huterms,hcterms,pbfs,spfs):
    """Computes all the individual operators that act on the spfs
    """
    # compute uncorrelated op*spfs
    if len(huterms) == 0:
        uopspfs = None
    else:
        uopspfs = np.zeros(nel, dtype=np.ndarray)
        for alpha in range(nel):
            uopspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
        # loop over all the uncorrelated terms in the hamiltonian
        for hterm in huterms:
            for alpha in range(nel):
                # does this term act on this electronic state?
                elop = hterm['elop']
                if elop_state(alpha,elop) != 0:
                    mel = compute_el_mes(alpha,alpha,elop)
                    opspfs = uopspfs[alpha]
                    coeff = hterm['coeff']
                    if 'modes' in hterm:
                        # act it on the mode
                        mode  = hterm['modes'][0]
                        nspf  = nspfs[alpha,mode]
                        npbf  = npbfs[mode]
                        ind   = spfstart[alpha,mode]
                        op    = hterm['ops'][0]
                        for n in range(nspf):
                            spf = spfs[alpha][ind:ind+npbf]
                            opspfs[ind:ind+npbf] += coeff*mel*pbfs[mode].operate(spf,op)
                            ind += npbf
                    else:
                        # multiply coefficient to spfs
                        opspfs += coeff*mel*spfs[alpha]
    # compute correlated op*spfs
    if len(hcterms) == 0:
        copspfs = None
    else:
        copspfs = np.zeros(len(hcterms), dtype=np.ndarray)
        # loop over all the correlated terms in the hamiltonian
        for i,hterm in enumerate(hcterms):
            # loop over each electronic state
            opspfs = np.zeros(nel, dtype=np.ndarray)
            for alpha in range(nel):
                # does this term act on this electronic state?
                elop = hterm['elop']
                if elop_state(alpha,elop) != 0:
                    opspfs[alpha] = np.zeros_like(spfs[alpha], dtype=complex)
                    coeff = hterm['coeff']
                    # act it on the modes
                    modes = hterm['modes']
                    for j,mode in enumerate(modes):
                        nspf = nspfs[alpha,mode]
                        npbf = npbfs[mode]
                        ind  = spfstart[alpha,mode]
                        op   = hterm['ops'][j]
                        for n in range(nspf):
                            spf = spfs[alpha][ind:ind+npbf]
                            opspfs[alpha][ind:ind+npbf] += pbfs[mode].operate(spf,op)
                            ind += npbf
            copspfs[i] = opspfs
    return uopspfs,copspfs

def compute_opips(nel,nmodes,nspfs,npbfs,spfstart,spfend,uopspfs,copspfs,
                  hcterms,spfs):
    """Computes all the individual operators innerproducts
    """
    # compute inner products for uncorrelated terms
    if uopspfs is None:
        uopips = None
    else:
        uopips = np.zeros(nel, dtype=np.ndarray)
        for alpha in range(nel):
            uopipsmode = np.zeros(nmodes, dtype=np.ndarray)
            for mode in range(nmodes):
                nspf = nspfs[alpha,mode]
                npbf = npbfs[mode]
                ind0 = spfstart[alpha,mode]
                indf = spfend[alpha,mode]
                spf = spfs[alpha][ind0:indf]
                opspf = uopspfs[alpha][ind0:indf]
                uopipsmode[mode] = spf_innerprod(nspf,nspf,npbf,spf,opspf)
            uopips[alpha] = uopipsmode
    # compute inner products for correlated terms
    copips = None
    if copspfs is None:
        copips = None
    else:
        copips = []
        for i,hcterm in enumerate(hcterms):
            # get opspfs
            opspfs = copspfs[i]
            # get info about this term
            modes = hcterm['modes']
            elop  = hcterm['elop']
            coeff = hcterm['coeff']
            # make dictionary for opips of this term
            opips = {}
            opips['modes'] = hcterm['modes']
            opips['elop']  = elop
            opips['coeff'] = coeff
            opip = []
            for alpha in range(nel):
                opip_a = []
                for beta in range(alpha,nel):
                    opip_b = []
                    mel = compute_el_mes(alpha,beta,elop)
                    if mel != 0.0:
                        for mode in modes:
                            nspf_l = nspfs[alpha,mode]
                            nspf_r = nspfs[beta,mode]
                            npbf   = npbfs[mode]
                            ind0_l = spfstart[alpha,mode]
                            indf_l = spfend[alpha,mode]
                            ind0_r = spfstart[beta,mode]
                            indf_r = spfend[beta,mode]
                            spf    = spfs[alpha][ind0_l:indf_l]
                            opspf  = copspfs[i][beta][ind0_r:indf_r]
                            opip_b.append( spf_innerprod(nspf_l,nspf_r,npbf,spf,opspf) )
                    opip_a.append( opip_b )
                opip.append( opip_a )
            opips['opips'] = opip
            copips.append( opips )
    return uopips,copips

# TODO this is not very general
def opmatel(nel,nmodes,uopips,spfovs,A,A_):
    """
    """
    for alpha in range(nel):
        for mode in range(nmodes):
            A_[alpha] += matelcontract(nmodes,[mode],[uopips[alpha][mode]],
                                       A[alpha],spfovs=spfovs[alpha][0])

def umatel(nel,nmodes,uopips,A,A_):
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products for the uncorrelated terms in the Hamiltonian.
    """
    for alpha in range(nel):
        for mode in range(nmodes):
            #if A_[alpha] is not 0:
            #    A_[alpha] += matelcontract(nmodes,[mode],[uopips[alpha][mode]],A[alpha])
            #else:
            #    A_[alpha] = matelcontract(nmodes,[mode],[uopips[alpha][mode]],A[alpha])
            A_[alpha] += matelcontract(nmodes,[mode],[uopips[alpha][mode]],A[alpha])

def uelmatel(nel,nmodes,huelterms,A,A_):
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products for the uncorrelated terms in the Hamiltonian.
    """
    for i,hterm in enumerate(huelterms):
        coeff = hterm['coeff']
        elop  = hterm['elop']
        for alpha in range(nel):
            mel = compute_el_mes(alpha,alpha,elop)
            if mel != 0.0:
                #if A_[alpha] is not 0:
                #    A_[alpha] += mel*coeff*A[alpha]
                #else:
                #    A_[alpha] = mel*coeff*A[alpha]
                A_[alpha] += mel*coeff*A[alpha]

def cmatel(nel,nmodes,hcelterms,copips,spfovs,A,A_):
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products for the correlated terms in the Hamiltonian.
    """
    for i,cip in enumerate(copips):
        modes = cip['modes']
        elop  = cip['elop']
        opips = cip['opips']
        coeff = cip['coeff']
        for alpha in range(nel):
            for beta in range(nel):
                mel = compute_el_mes(alpha,beta,elop)
                if mel != 0.0:
                    mel *= coeff
                    if alpha == beta:
                        opip = opips[alpha][alpha]
                        ovs  = None
                        conj = False
                    elif alpha < beta:
                        opip = opips[alpha][beta]
                        ovs = spfovs[alpha][beta-alpha-1]
                        conj = False
                    else:
                        opip = opips[beta][alpha]
                        ovs = spfovs[beta][alpha-beta-1]
                        conj = True
                    #if A_[alpha] is not 0:
                    #    A_[alpha] += mel*matelcontract(nmodes,modes,opip,A[beta],
                    #                               spfovs=ovs,conj=conj)
                    #else:
                    #    A_[alpha] = mel*matelcontract(nmodes,modes,opip,A[beta],
                    #                               spfovs=ovs,conj=conj)
                    A_[alpha] += mel*matelcontract(nmodes,modes,opip,A[beta],
                                               spfovs=ovs,conj=conj)

def celmatel(nel,nmodes,hcelterms,spfovs,A,A_):
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products for the correlated terms in the Hamiltonian.
    """
    for i,hterm in enumerate(hcelterms):
        for alpha in range(nel):
            for beta in range(nel):
                mel = compute_el_mes(alpha,beta,hterm['elop'])
                if mel != 0.0:
                    mel *= hterm['coeff']
                    if alpha < beta:
                        ovs  = spfovs[alpha][beta-alpha-1]
                        conj = False
                    else:
                        ovs  = spfovs[beta][alpha-beta-1]
                        conj = True
                    #if A_[alpha] is not 0:
                    #    A_[alpha] += mel*matelcontract(nmodes,None,None,A[beta],
                    #                         spfovs=ovs,conj=conj)
                    #else:
                    #    A_[alpha] = mel*matelcontract(nmodes,None,None,A[beta],
                    #                         spfovs=ovs,conj=conj)
                    A_[alpha] += mel*matelcontract(nmodes,None,None,A[beta],
                                         spfovs=ovs,conj=conj)


def matel(nel,nmodes,nspfs,npbfs,uopips,copips,huelterms,hcelterms,spfovs,A):
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products.
    """
    # make the output A tensor
    Aout = np.zeros(nel, dtype=np.ndarray)
    for alpha in range(nel):
        Aout[alpha] = np.zeros_like(A[alpha], dtype=complex)
    # compute matrix elements for uncorrelated terms
    if not uopips is None:
        umatel(nel,nmodes,uopips,A,Aout)
    # compute matrix elements for correlated terms
    if not copips is None:
        cmatel(nel,nmodes,hcelterms,copips,spfovs,A,Aout)
    # compute matrix elements for uncorrelated electronic terms
    if len(huelterms) != 0:
        uelmatel(nel,nmodes,huelterms,A,Aout)
    # compute matrix elements for correlated electronic terms
    if len(hcelterms) != 0:
        celmatel(nel,nmodes,hcelterms,spfovs,A,Aout)
    return Aout

# TODO need to generalize this to many-mode operators
def act_operator(A,spfs,wf,op,pbfs,psi,tol=1.e-8,maxiter=100):
#def act_operator(A,spfs,wf,op,pbfs,tol=1.e-8,maxiter=100):
    """
    """
    from pymctdh.meanfield import compute_meanfield_uncorr_op
    # get wf info
    nel      = wf.nel
    nmodes   = wf.nmodes
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend
    psistart = wf.psistart
    psiend   = wf.psiend

    # create wavefunction data for iterations
    A_k      = np.zeros(nel, dtype=np.ndarray)
    A_kp1    = np.zeros(nel, dtype=np.ndarray)
    spfs_k   = np.zeros(nel, dtype=np.ndarray)
    spfs_kp1 = np.zeros(nel, dtype=np.ndarray)
    for alpha in range(nel):
        A_k[alpha]      = np.zeros_like(A[alpha], dtype=complex)
        A_kp1[alpha]    = np.zeros_like(A[alpha], dtype=complex)
        spfs_k[alpha]   = np.zeros_like(spfs[alpha], dtype=complex)
        spfs_kp1[alpha] = np.zeros_like(spfs[alpha], dtype=complex)

    # compute action of operator on wf spfs
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                        spfstart,spfend,[op.term],[],pbfs,spfs)

    # compute overlap matrices
    spfovs = overlap_matrices2(nel,nmodes,nspfs,npbfs,spfstart,spfs,spfs)

    # update trial A coefficients
    opmatel(nel,nmodes,uopips,spfovs,A,A_k)

    # perform iterations until convergence is reached
    flag = 1
    for i in range(maxiter):
        # update spfs
        compute_meanfield_uncorr_op(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                    op.term,uopspfs,spfs,spfs_kp1)
        # gram-schmidt orthogonalize the spfs
        spfs_kp1 = wf.orthonormalize_spfs(spfs_kp1)
        # compute new matrix elements
        uopips,copips = compute_opips(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                            uopspfs,copspfs,[],spfs_kp1)
        spfovs = overlap_matrices2(nel,nmodes,nspfs,npbfs,spfstart,spfs_kp1,spfs)
        for alpha in range(nel):
            A_kp1[alpha] *= 0.0
        # update trial A coefficients
        opmatel(nel,nmodes,uopips,spfovs,A,A_kp1)
        # check convergence
        delta = 0.0
        for alpha in range(nel):
            for mode in range(nmodes):
                ind0 = spfstart[alpha,mode]
                indf = spfend[alpha,mode]
                # compute criteria based on projectors
                p_k = compute_projector(nspfs[alpha,mode],npbfs[mode],spfs_k[alpha][ind0:indf])
                p_kp1 = compute_projector(nspfs[alpha,mode],npbfs[mode],spfs_kp1[alpha][ind0:indf])
                delta += np.linalg.norm(p_kp1-p_k)
        if delta < tol:
            flag = 0
            break
        else:
            A_k    = deepcopy(A_kp1)
            spfs_k = deepcopy(spfs_kp1)

    if flag:
        raise ValueError("Maximum iterations reached. Could not converge.")

    # reshape everything for output
    for alpha in range(nel):
        ind0 = psistart[0,alpha]
        indf = psiend[0,alpha]
        psi[ind0:indf] = A_kp1[alpha].ravel()
        #wf.psi[ind0:indf] = A_kp1[alpha].ravel()
        ind0 = psistart[1,alpha]
        indf = psiend[1,alpha]
        psi[ind0:indf] = spfs_kp1[alpha]
        #wf.psi[ind0:indf] = spfs_kp1[alpha]
    #wf.normalize()

if __name__ == "__main__":

    from time import time
    from pbasis import PBasis
    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from qoperator import QOperator
    #from eom import eom_coeffs
    #from meanfield import compute_meanfield_uncorr
    from cy.wftools import overlap_matrices
    import cProfile
    import old_stuff.optools

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

    ### test of uopspfs and uopips ###
    print('testing uncorrelated opspfs and opips')
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,wf.spfstart,wf.spfend,ham.huterms,ham.hcterms,pbfs,wf.spfs)
    opspfs,opips = old_stuff.optools.precompute_ops(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfend,ham.ops,pbfs,wf.spfs)
    for alpha in range(nel):
        print('el state')
        mode = 0
        print('mode',mode)
        nspf = wf.nspfs[alpha,mode]
        npbf = wf.npbfs[mode]
        ind0 = wf.spfstart[alpha,mode]
        indf = wf.spfend[alpha,mode]
        spf  = hterms[1]['coeff']*opspfs[alpha][mode]['KE']
        spf += hterms[2]['coeff']*opspfs[alpha][mode]['q^2']
        if alpha == 0:
            spf += hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
        else:
            spf -= hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
        opip = spf_innerprod(nspf,nspf,npbf,wf.spfs[alpha][ind0:indf],spf)
        print(np.allclose(opip,uopips[alpha][mode]))
        print(np.allclose(spf,uopspfs[alpha][ind0:indf]))
        mode = 1
        print('mode',mode)
        nspf = wf.nspfs[alpha,mode]
        npbf = wf.npbfs[mode]
        ind0 = wf.spfstart[alpha,mode]
        indf = wf.spfend[alpha,mode]
        spf  = hterms[3]['coeff']*opspfs[alpha][mode]['KE']
        spf += hterms[4]['coeff']*opspfs[alpha][mode]['q^2']
        if alpha == 0:
            spf += hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[10]['coeff']*opspfs[alpha][mode]['q']
        else:
            spf -= hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[11]['coeff']*opspfs[alpha][mode]['q']
        opip = spf_innerprod(nspf,nspf,npbf,wf.spfs[alpha][ind0:indf],spf)
        print(np.allclose(opip,uopips[alpha][mode]))
        print(np.allclose(spf,uopspfs[alpha][ind0:indf]))
        mode = 2
        print('mode',mode)
        nspf = wf.nspfs[alpha,mode]
        npbf = wf.npbfs[mode]
        ind0 = wf.spfstart[alpha,mode]
        indf = wf.spfend[alpha,mode]
        spf  = hterms[5]['coeff']*opspfs[alpha][mode]['KE']
        spf += hterms[6]['coeff']*opspfs[alpha][mode]['q^2']
        if alpha == 0:
            spf += hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[12]['coeff']*opspfs[alpha][mode]['q']
        else:
            spf -= hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[13]['coeff']*opspfs[alpha][mode]['q']
        opip = spf_innerprod(nspf,nspf,npbf,wf.spfs[alpha][ind0:indf],spf)
        print(np.allclose(opip,uopips[alpha][mode]))
        print(np.allclose(spf,uopspfs[alpha][ind0:indf]))
        mode = 3
        print('mode',mode)
        nspf = wf.nspfs[alpha,mode]
        npbf = wf.npbfs[mode]
        ind0 = wf.spfstart[alpha,mode]
        indf = wf.spfend[alpha,mode]
        spf  = hterms[7]['coeff']*opspfs[alpha][mode]['KE']
        spf += hterms[8]['coeff']*opspfs[alpha][mode]['q^2']
        if alpha == 0:
            spf += hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[14]['coeff']*opspfs[alpha][mode]['q']
        else:
            spf -= hterms[0]['coeff']*wf.spfs[alpha][ind0:indf]
            spf += hterms[15]['coeff']*opspfs[alpha][mode]['q']
        opip = spf_innerprod(nspf,nspf,npbf,wf.spfs[alpha][ind0:indf],spf)
        print(np.allclose(opip,uopips[alpha][mode]))
        print(np.allclose(spf,uopspfs[alpha][ind0:indf]))

    ### test of compute copspfs to old opspfs ###
    print('testing correlated opspfs and opips')
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,wf.spfstart,wf.spfend,ham.huterms,ham.hcterms,pbfs,wf.spfs)
    opspfs,opips = old_stuff.optools.precompute_ops(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.spfend,ham.ops,pbfs,wf.spfs)
    ind0 = wf.spfstart[0,0]
    indf = wf.spfend[0,0]
    print(np.allclose(copspfs[0][0][ind0:indf],hterms[9]['coeff']*opspfs[0][0]['q']))
    ind0 = wf.spfstart[1,0]
    indf = wf.spfend[1,0]
    print(np.allclose(copspfs[0][1][ind0:indf],hterms[9]['coeff']*opspfs[1][0]['q']))
    print(np.allclose(copips[0]['opips'][0][1][0],hterms[9]['coeff']*opips[0][1][0]['q']))
