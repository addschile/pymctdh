import numpy as np
from pymctdh.optools import precompute_ops
from pymctdh.cy.tensorutils import atensorcontract
from pymctdh.cy.wftools import reshape_wf

def compute_expect(nel, nmodes, nspfs, npbfs, spfstart, spfend, psistart,
                   psiend, psi, op, pbfs):
    """Computes the expectation value of a generic operator.
    NOTE: only single-mode operators is supported.
    """

    A,spfs = reshape_wf(nel,nmodes,nspfs,psistart,psiend,psi)

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

def diabatic_pops(nel,psistart,psiend,psi):
    """Computes diabatic populations for each electronic state.
    """
    # compute diabatic populations
    pops = np.zeros(nel)
    for i in range(nel):
        ind0 = psistart[0,i]
        indf = psiend[0,i]
        pops[i] = np.sum(psi[ind0:indf].conj()*psi[ind0:indf]).real
    return pops

def diabatic_grid_pops(nel,nmodes,nspfs,npbfs,psistart,psiend,spfstart,spfend,
                       pbfs,psi,modes=None):
    """Function that computes populations at each grid point in diabatic states.
    """
    # check and make sure projectors are made for combined modes
    for i in range(len(pbfs)):
        if pbfs[i].combined and pbfs[i].dvrproj is None:
            pbfs[i].makedvrproj()

    # get total number of modes in simulation
    tot_nmodes = 0
    for i in range(nmodes):
        tot_nmodes += pbfs[i].nmodes

    A,spfs = reshape_wf(nel,nmodes,nspfs,psistart,psiend,psi)

    if modes is None:
        grid_pops = np.zeros(tot_nmodes, dtype=np.ndarray)
        modes = [i for i in range(tot_nmodes)]
    else:
        grid_pops = np.zeros(len(modes), dtype=np.ndarray)

    modecount = 0
    modetrack = 0
    for mode in range(nmodes):
        pbf  = pbfs[mode]
        if pbf.combined == True:
            raise NotImplementedError
        else:
            if modetrack in modes:
                npbf = pbf.params['npbf']
                pops = np.zeros((nel,npbf))
                for alpha in range(nel):
                    ind0 = spfstart[alpha,mode]
                    indf = spfend[alpha,mode]
                    # make single hole function
                    G = pymctdh.cy.tensorutils.atensorcontract(nmodes,[mode],A[alpha])
                    # transform spfs for this mode to dvr basis
                    if not pbf.params['dvr']:
                        spf = np.zeros_like(spfs[alpha][ind0:indf])
                        ind00 = 0
                        for i in range(nspfs[alpha,mode]):
                            indf  = ind0 + npbf
                            indff = ind00 + npbf
                            spf[ind00:indff] = pbf.dvrtrans(spfs[alpha][ind0:indf])
                            ind0 += npbf
                            ind00 += npbf
                    else:
                        spf = spfs[alpha][ind0:indf]
                    # compute inner products between spfs
                    ind0l = 0 
                    for i in range(nspfs[alpha,mode]):
                        indfl = ind0l + npbf
                        spf_l = spf[ind0l:indfl]
                        ind0r = 0
                        for j in range(nspfs[alpha,mode]):
                            indfr = ind0r + npbf
                            spf_r = spf[ind0r:indfr]
                            pops[alpha,:] += (G[i,j]*spf_l.conj()*spf_r).real
                            ind0r += npbf
                        ind0l += npbf
                    ind0 = ind0l
                    grid_pops[modecount] = pops
                modecount += 1
            modetrack += 1
    return grid_pops

#TODO
#def adiabatic_pops(wf):
#    """Computes diabatic populations for each electronic state.
#    """
#    # get wf info
#    nel      = wf.nel
#    psistart = wf.psistart
#    psiend   = wf.psiend
#    psi      = wf.psi
#    # compute diabatic populations
#    pops = np.zeros(nel)
#    for i in range(self.nel):
#        ind0 = psistart[0,i]
#        indf = psiend[0,i]
#        pops[i] = np.sum(psi[ind0:indf].conj()*psi[ind0:indf]).real
#    return pops
