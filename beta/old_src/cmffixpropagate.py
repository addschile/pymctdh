import sys
from copy import deepcopy
from time import time

import numpy as np
import scipy.integrate

from krylov import krylov_prop,krylov_prop_ada
from eom import cmfeom_spfs
from optools import precompute_ops
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,compute_projector)
from meanfield import compute_meanfield_mats

def cmfprecomputemels(nel, nmodes, nspfs, npbfs, spfstart, spfend, ham, pbfs, spfsin):
    """
    """
    # reshape spfs
    spfs = np.zeros(nel, dtype=np.ndarray)
    for i in range(nel):
        ind0 = spfstart[i,0]
        indf = spfend[i,-1]
        if i!=0:
            ind0 += spfend[i-1,-1]
            indf += spfend[i-1,-1]
        spfs[i] = spfsin[ind0:indf]

    # precompute stuff for propagation
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                        spfstart,spfend,ham.huterms,ham.hcterms,
                                        pbfs,spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,spfs)

    return uopspfs,copspfs,uopips,copips,spfovs

def cmfprecomputemf(nel, nmodes, nspfs, npbfs, spfstart, spfend, ham, copips,
                    spfovs, spfs, A):
    """
    """
    # compute mean-field matrices
    mfs = None
    if copips is not None:
        mfs = compute_meanfield_mats(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                     copips,spfovs,A)

    # compute density matrices and projectors
    rhos  = []
    #projs = []
    for alpha in range(nel):
        rho_tmp = []
        #proj_tmp = []
        for mode in range(nmodes):
            nspf = nspfs[alpha,mode]
            npbf = npbfs[mode]
            ind0 = spfstart[alpha,mode]
            indf = spfend[alpha,mode]
            if alpha!=0:
                ind0 += spfend[alpha-1,-1]
                indf += spfend[alpha-1,-1]
            # compute and invert density matrix
            rho = compute_density_matrix(nspf,alpha,nmodes,mode,A[alpha])
            rho_tmp.append( invert_density_matrix(rho) )
            #proj_tmp.append( compute_projector(nspf,npbf,spfs[ind0:indf]) )
        rhos.append( rho_tmp )
        #projs.append( proj_tmp )

    return mfs,rhos#,projs

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

    ## get A tensor for coeffs ##
    indf = psiend[0,-1]
    A = np.zeros(2, dtype=np.ndarray)
    for alpha in range(nel):
        shaper = ()
        for mode in range(nmodes):
            shaper += (nspfs[alpha,mode],)
        ind0 = psistart[0,alpha] 
        indf = psiend[0,alpha]
        A[alpha] = np.reshape(wf.psi[ind0:indf], shaper, order='C')

    ## integrator for spfs ##
    rspfs = scipy.integrate.ode(cmfeom_spfs)
    rspfs.set_integrator('zvode',method='adams',order=12,atol=1.e-8,rtol=1.e-6,
                         nsteps=1000,first_step=0,min_step=0,max_step=dt)
    # set parameters for integrator function
    ind     = psistart[1,0]
    npsis   = len(wf.psi[ind:])
    uopspfs = None
    copspfs = None
    uopips  = None
    copips  = None
    spfovs  = None
    mfs     = None
    rhos    = None
    projs   = None
    ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A,mfs,rhos]
    rspfs.set_f_params(*ode_spfs_args)
    # set initial condition
    ind = wf.psistart[1,0]
    rspfs.set_initial_value(wf.psi[ind:], times[0])

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
        #if i==0:
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                   nspfs,npbfs,spfstart,spfend,
                                                   ham,pbfs,rspfs.y)
        #mfs,rhos,projs = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,
        #                    ham,copips,spfovs,rspfs.y,A)
        mfs,rhos = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,
                                   copips,spfovs,rspfs.y,A)
        #print('mfs')
        #print(mfs)
        #print('rhos')
        #print(rhos)
        #raise ValueError

        # reset ode f params for spfs
        #ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham.huelterms,
        #                 ham.hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A.copy(),mfs,
        #                 rhos,projs]
        ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,
                         A.copy(),copspfs,copips,spfovs,mfs,rhos]
        #ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A.copy(),mfs,rhos]
        rspfs.set_f_params(*ode_spfs_args)

        # integrate coefficients one half timestep forward
        A = krylov_prop(A,0.5*dt,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,
                        method='lanczos',return_all=False)
        #A = krylov_prop_ada(times[i],times[i]+0.5*dt,0.5*dt,A,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,
        #                method='lanczos',return_all=False)
        # integrate spfs one half timestep forward
        spfstmp = rspfs.y.copy()
        rspfs.integrate(rspfs.t+0.5*dt/5.)
        #for j in range(5):
        #    rspfs.integrate(rspfs.t+0.5*dt/5.)#,relax=True)

        # compute matrix elements and meanfield matrices
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                   nspfs,npbfs,spfstart,spfend,
                                                   ham,pbfs,rspfs.y)
        #mfs,rhos,projs = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,
        #                    ham,copips,spfovs,rspfs.y,A)
        mfs,rhos = cmfprecomputemf(nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,
                                   copips,spfovs,rspfs.y,A)

        # reset ode f params for spfs
        #ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham.huelterms,
        #                 ham.hcelterms,uopspfs,copspfs,uopips,copips,spfovs,A,mfs,
        #                 rhos,projs]
        ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,
                         A.copy(),copspfs,copips,spfovs,mfs,rhos]
        #ode_spfs_args = [npsis,nel,nmodes,nspfs,npbfs,spfstart,spfend,ham,pbfs,A.copy(),mfs,rhos]
        rspfs.set_f_params(*ode_spfs_args)

        # integrate spfs one half timestep forward
        rspfs.set_initial_value(spfstmp, times[i])
        rspfs.integrate(rspfs.t+dt)
        #for j in range(10):
        #    rspfs.integrate(rspfs.t+dt/10.)#,relax=True)

        # compute matrix elements
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(nel,nmodes,
                                                   nspfs,npbfs,spfstart,spfend,
                                                   ham,pbfs,rspfs.y)

        # integrate coefficients one half timestep forward
        #print((times[i]+0.5*dt)/41.3413745758)
        A = krylov_prop(A,0.5*dt,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,
                        method='lanczos',return_all=False)
        #A = krylov_prop_ada(times[i]+0.5*dt,times[i+1],0.5*dt,A,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,
        #                method='lanczos',return_all=False)

        if i%1==0:
            # reset wf
            for alpha in range(nel):
                ind0 = psistart[0,alpha]
                indf = psiend[0,alpha]
                wf.psi[ind0:indf] = A[alpha].ravel()
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
