import numpy as np
from optools import precompute_ops
from cy.tensorutils import atensorcontract
#from cy.wftools import spf_innerprod,overlap_matrices2,compute_projector

# TODO this also needs to be generalized to many-mode operators
def compute_expect(op,wf,pbfs):
    """Computes the expectation value of a generic operator.
    """
    # get wf info
    nmodes   = wf.nmodes
    nel      = wf.nel
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend
    psistart = wf.psistart
    psiend   = wf.psiend
    psi      = wf.psi

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
        A[alpha] = np.reshape(wf.psi[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = wf.psistart[1,alpha]
        indf = wf.psiend[1,alpha]
        spfs[alpha] = wf.psi[ind0:indf]

    modes = op.term['modes']
    mode = op.term['modes'][0]
    # make matrix of spf inner products
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                        spfstart,spfend,[op.term],[],pbfs,spfs)
    # contract (A*)xA with opips
    expect = 0.0
    for alpha in range(nel):
        Asum = atensorcontract(nmodes, modes, A[alpha])
        expect += np.tensordot(Asum,uopips[alpha][mode],axes=[[0,1],[0,1]]).real

    return expect

def diabatic_pops(wf):
    """Computes diabatic populations for each electronic state.
    """
    # get wf info
    nel      = wf.nel
    psistart = wf.psistart
    psiend   = wf.psiend
    psi      = wf.psi
    # compute diabatic populations
    pops = np.zeros(nel)
    for i in range(self.nel):
        ind0 = psistart[0,i]
        indf = psiend[0,i]
        pops[i] = np.sum(psi[ind0:indf].conj()*psi[ind0:indf]).real
    return pops

## TODO
#def compute_expect(op,A,spfs,wf,pbfs):
#def avg_q(mode, nel=None):
#    """Computes average position of a mode
#    """
#    if nel==None:
#        # compute average on both surfaces
#        for alpha in range(nel):
#            Asum = tensorcontract([mode],wf.A[alpha,:])
#        for i in range(self.nspf[mode]):
#            
#    else:
#    pops = np.zeros(self.nel)
#    for i in range(self.nel):
#        pops[i] = np.sum(self.A[i,:].conj()*self.A[i,:]).real
#    return pops
