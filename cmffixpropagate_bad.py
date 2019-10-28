import numpy as np
import scipy.integrate
from krylov import krylov_prop
from eom import cmfeom_spfs
from optools import precompute_ops
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,compute_projector)
from meanfield import compute_meanfield_mats
from copy import deepcopy
from time import time
import sys

def cmfprecomputemels(nel, nmodes, nspfs, npbfs, spfstart, spfend, ham, pbfs, spfsin):
    """
    """
    # reshape spfs
    spfs = np.zeros(nel, dtype=np.ndarray)
    for i in range(nel):
        ind0 = spfstart[i,0]
        indf = spfend[i,-1]
        spfs[i] = spfsin[ind0:indf]

    # precompute stuff for propagation
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                        spfstart,spfend,ham.huterms,ham.hcterms,
                                        pbfs,spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)

    return uopspfs,copspfs,uopips,copips,spfovs

def cmfprecomputemf(nel, nmodes, nspfs, npbfs, spfstart, spfend, ham, copips,
                    spfovs, spfsin, A):
    """
    """
    # reshape spfs
    spfs = np.zeros(nel, dtype=np.ndarray)
    for i in range(nel):
        ind0 = spfstart[i,0]
        indf = spfend[i,-1]
        spfs[i] = spfsin[ind0:indf]

    # compute mean-field matrices
    mfs = None
    if copips is not None:
        mfs = compute_meanfield_mats(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                     copips,spfovs,A)

    # compute density matrices and projectors
    rhos  = []
    projs = []
    for alpha in range(nel):
        rho_tmp = []
        proj_tmp = []
        for mode in range(nmodes):
            nspf = nspfs[alpha,mode]
            npbf = npbfs[mode]
            ind0 = spfstart[alpha,mode]
            indf = spfend[alpha,mode]
            # compute and invert density matrix
            rho = compute_density_matrix(nspf,alpha,nmodes,mode,A[alpha])
            rho_tmp.append( invert_density_matrix(rho) )
            # compute projector
            proj_tmp.append( compute_projector(nspf,npbf,spfs[alpha][ind0:indf]) )
        rhos.append( rho_tmp )
        projs.append( proj_tmp )

    return mfs,rhos,projs

def cmffixpropagate(times, ham, pbfs, wf, filename):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the constant mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """
    # get wavefunction info
    nel      = wf.nel
    nmodes   = wf.nmodes
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend
    psistart = wf.psistart
    psiend   = wf.psiend

    ### set up integrator and options ###
    dt = times[1]-times[0]

    ## integrator for coeffs ##
    rcoeffs = krylov_prop
    indf = psiend[0,-1]
    A = np.zeros(2, dtype=np.ndarray)
    for alpha in range(nel):
        shaper = ()
        for mode in range(nmodes):
            shaper += (nspfs[alpha,mode],)
        ind0 = psistart[0,alpha] 
        indf = psiend[0,alpha]
        A[alpha] = np.reshape(wf.psi[ind0:indf], shaper, order='C')
    # NOTE can't use two instances of scipy ZVODE
    #rcoeffs = scipy.integrate.ode(cmfeom_coeffs)
    #rcoeffs.set_integrator('zvode',method='adams',order=12,atol=1.e-8,rtol=1.e-6,
    #                       nsteps=1000,first_step=0,min_step=0,max_step=dt)
    ## set parameters for integrator function
    #ind = psiend[0,-1]
    #npsic = len(wf.psi[:ind])
    #uopips = None
    #copips = None
    #spfovs = None
    #ode_coeffs_args = [npsic,nel,nmodes,nspfs,npbfs,psistart,psiend,ham.huelterms,
    #                   ham.hcelterms,uopips,copips,spfovs]
    #rcoeffs.set_f_params(*ode_coeffs_args)
    ## set initial condition
    #indf = psiend[0,-1]
    #rcoeffs.set_initial_value(wf.psi[:indf], times[0])

    ## integrator for spfs ##
    rspfs = scipy.integrate.ode(cmfeom_spfs)
    rspfs.set_integrator('zvode',method='adams',order=12,atol=1.e-8,rtol=1.e-6,
                         nsteps=1000,first_step=0,min_step=0,max_step=dt)
    # set parameters for integrator function
    ind     = psiend[1,0]
    npsis   = len(wf.psi[ind:])
    uopspfs = None
    copspfs = None
    uopips  = None
    copips  = None
    spfovs  = None
    mfs     = None
    rhos    = None
    projs   = None
    ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham.huelterms,
                     ham.hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A,mfs,
                     rhos,projs]
    rspfs.set_f_params(*ode_spfs_args)
    # set initial condition
    ind0 = wf.psistart[1,0]
    indf = wf.psistart[1,-1]
    rspfs.set_initial_value(wf.psi[ind0:indf], times[0])

    f = open(filename,'w')
    every = int(len(times)/10)
    btime = time()
    for i in range(len(times)-1):
        if i%every==0:
            sys.stdout.write("%.0f Percent done"%(10*(i/every))+"."*10+"%.8f\n"%(time()-btime))
            sys.stdout.flush()
        # compute any expectation values
        if i%1==0:
            pops = wf.diabatic_pops()
            f.write('%.8f '%(times[i]))
            for j in range(len(pops)):
                f.write('%.8f '%(pops[j]))
        # compute matrix elements and meanfield matrices
        #A = np.zeros(2, dtype=np.ndarray)
        #for alpha in range(nel):
        #    shaper = ()
        #    for mode in range(nmodes):
        #        shaper += (nspfs[alpha,mode],)
        #    ind0 = psistart[0,alpha] 
        #    indf = psiend[0,alpha]
        #    A[alpha] = np.reshape(rcoeffsy[ind0:indf], shaper, order='C')
        if i==0:
            uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                       nspfs,npbfs,spfstart,spfend,
                                                       ham,pbfs,rspfs.y)
        mfs,rhos,projs = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                            ham,copips,spfovs,rspfs.y,A)

        # reset ode f params
        # coeffs
        #rcoeffs.f_params[-3:] = [uopips,copips,spfovs]
        #ode_coeffs_args = [npsic,nel,nmodes,nspfs,npbfs,psistart,psiend,ham.huelterms,
        #                   ham.hcelterms,uopips,copips,spfovs]
        #rcoeffs.set_f_params(*ode_coeffs_args)
        # spfs
        #rspfs.f_params[-9:] = [uopspfs,copspfs,uopips,copips,spfovs,A,mfs,rhos,projs]
        ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham.huelterms,
                         ham.hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A,mfs,
                         rhos,projs]
        rspfs.set_f_params(*ode_spfs_args)

        # integrate coefficients one half timestep forward
def krylov_prop(t, dt, y, npsi, nel, nmodes, nspfs, npbfs, psistart, psiend, ham,
                uopips, copips, spfovs, method='lanczos', return_all=False):
        rcoeffs(times[i],0.5*dt,A,npsic,nel,nmodes,n
        #rcoeffs.integrate(rcoeffs.t+0.5*dt)
        # integrate spfs one half timestep forward
        spfstmp = rspfs.y.copy()
        rspfs.integrate(rspfs.t+0.5*dt)

        # compute matrix elements and meanfield matrices
        A = np.zeros(2, dtype=np.ndarray)
        for alpha in range(nel):
            shaper = ()
            for mode in range(nmodes):
                shaper += (nspfs[alpha,mode],)
            ind0 = psistart[0,alpha] 
            indf = psiend[0,alpha]
            A[alpha] = np.reshape(rcoeffsy[ind0:indf], shaper, order='C')
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                   nspfs,npbfs,spfstart,spfend,
                                                   ham,pbfs,rspfs.y)
        mfs,rhos,projs = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                            ham,copips,spfovs,rspfs.y,A)

        # reset ode f params for spfs
        #rspfs.f_params[-9:] = [uopspfs,copspfs,uopips,copips,spfovs,A,mfs,rhos,projs]
        ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham.huelterms,
                         ham.hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A,mfs,
                         rhos,projs]
        rspfs.set_f_params(*ode_spfs_args)

        # integrate spfs one half timestep forward
        rspfs.set_initial_value(spfstmp, times[i])
        rspfs.integrate(rspfs.t+dt)

        # compute matrix elements
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                   nspfs,npbfs,spfstart,spfend,
                                                   ham,pbfs,rspfs.y)

        # reset ode f params for coeffs
        #rcoeffs.f_params[-3:] = [uopips,copips,spfovs]
        ode_coeffs_args = [npsic,nel,nmodes,nspfs,npbfs,psistart,psiend,ham.huelterms,
                           ham.hcelterms,uopips,copips,spfovs]
        rcoeffs.set_f_params(*ode_coeffs_args)

        # integrate coefficients one half timestep forward
        rcoeffs.integrate(rcoeffs.t+0.5*dt)

        if i%1==0:
            # reset wf
            ind = psiend[0,-1]
            wf.psi[:ind] = rcoeffs.y
            ind = psistart[1,0]
            wf.psi[ind:] = rspfs.y
            # compute norm
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
    f.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    return wf
