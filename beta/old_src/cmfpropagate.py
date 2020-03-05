import numpy as np
import integrator
from optools import precompute_ops
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,compute_projector)
from meanfield import compute_meanfield_mats
from copy import deepcopy
from time import time
import sys

def cmfprecomputemels(ham, pbfs, wf):
    """
    """
    # get wavefunction info
    nel      = wf.nel
    nmodes   = wf.nmodes
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend

    # precompute stuff for propagation
    uopspfs,copspfs,uopips,copips = precompute_ops(nel,nmodes,nspfs,npbfs,
                                        spfstart,spfend,ham.huterms,ham.hcterms,
                                        pbfs,wf.spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,wf.spfs)

    return uopspfs,copspfs,uopips,copips,spfovs

def cmfprecomputemf(ham, copips, spfovs, wf):
    """
    """
    # get wavefunction info
    nel      = wf.nel
    nmodes   = wf.nmodes
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend

    # compute mean-field matrices
    mfs = None
    if copips is not None:
        mfs = compute_meanfield_mats(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                     copips,spfovs,wf.A)

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
            rho = compute_density_matrix(nspf,alpha,nmodes,mode,wf.A[alpha])
            rho_tmp.append( invert_density_matrix(rho) )
            # compute projector
            proj_tmp.append( compute_projector(nspf,npbf,wf.spfs[alpha][ind0:indf]) )
        rhos.append( rho_tmp )
        projs.append( proj_tmp )

    return mfs,rhos,projs

#def cmfpropagate(times, ham, pbfs, wf, filename):
#    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
#    principle. Uses the constant mean field scheme in which 
#
#    Inputs
#    ------
#    Outputs
#    -------
#    """
#    # get wavefunction info
#    nel      = wf.nel
#    nmodes   = wf.nmodes
#    nspfs    = wf.nspfs
#    npbfs    = wf.npbfs
#    spfstart = wf.spfstart
#    spfend   = wf.spfend
#
#    # set up integrator and options
#    dt = times[1]-times[0]
#    Aintegrate = integrator.krylov_prop
#    spfsintegrate = integrator.rk
#
#    f = open(filename,'w')
#    every = int(len(times)/10)
#    btime = time()
#    for i in range(len(times)-1):
#        if i%every==0:
#            sys.stdout.write("%.0f Percent done"%(10*(i/every))+"."*10+"%.8f\n"%(time()-btime))
#            sys.stdout.flush()
#        # compute any expectation values
#        if i%1==0:
#            pops = wf.diabatic_pops()
#            f.write('%.8f '%(times[i]))
#            for j in range(len(pops)):
#                f.write('%.8f '%(pops[j]))
#        if i==0:
#            opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
#        dt = times[i+1]-times[i]
#        t0 = times[i]
#        while abs(t0-times[i+1]) > 1.e-12:
#            #print('time',i)
#            #print(t0,times[i+1])
#            if (times[i+1]-t0) < dt:
#                # timestep is too big
#                dt = times[i+1] - t0
#            mfs,rhos,projs = cmfprecomputemf(ham,opspfs,opips,spfovs,wf)
#            # integrate coefficients one half timestep forward
#            Atmp = wf.copy('A')
#            Aintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, opips, spfovs)
#            # integrate spfs one half timestep forward
#            spfstmp = wf.copy('spfs')
#            spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs, 
#                          cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
#                          spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
#            # precompute new stuff for propagation
#            opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
#            mfs,rhos,projs = cmfprecomputemf(ham,opspfs,opips,spfovs,wf)
#            # integrate spfs one full timestep forward
#            wf.spfs = deepcopy(spfstmp)
#            #spfsintegrate(times[i], times[i], dt, wf, ham, pbfs, 
#            #              cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
#            #              spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
#            spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs, 
#                          cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
#                          spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
#            spfsintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, pbfs, 
#                          cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
#                          spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
#            #precompute new stuff for propagation
#            opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
#            # computing the error of the propagation
#            # integrate coefficients one half timestep backward
#            Atmp2 = wf.copy('A')
#            Aintegrate(times[i]+0.5*dt, times[i], -0.5*dt, wf, ham, opips, spfovs)
#            diff = 0.0
#            for j in range(nel):
#                diff += np.linalg.norm(Atmp[j]-wf.A[j])
#            tot = 0.0
#            for j in range(nel):
#                tot += np.sum(np.abs(wf.A[j]))
#            err = diff/tot
#            if err > 5.e-4:
#            #if False:
#                # step was rejected
#                print('rejected')
#                print(err)
#                # revert wavefunction back
#                wf.A    = deepcopy(Atmp)
#                wf.spfs = deepcopy(spfstmp)
#                # update timestep
#                dtold = dt
#                #dt *= 0.5
#                dt *= (5.e-4/err)
#                print(dtold,dt)
#                opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
#            else:
#                # step was accepted
#                #print('accepted')
#                # propagate final half step forward
#                wf.A = deepcopy(Atmp2)
#                Aintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, opips, spfovs)
#                # update time and timestep
#                t0 += dt
#                dt = times[i+1]-t0
#        if i%1==0:
#            norm = wf.norm()
#            f.write('%.8f\n'%(norm))
#            f.flush()
#    f.close()
#    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
#    sys.stdout.flush()
#    return wf

def cmffixpropagate(times, ham, pbfs, wf, filename, comperror=False):
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

    # set up integrator and options
    dt = times[1]-times[0]
    Aintegrate = integrator.krylov_prop
    spfsintegrate = integrator.rk

    if comperror:
        ferr = open('error_file.txt','w')
    f = open(filename,'w')
    every = int(len(times)/10)
    btime = time()
    for i in range(len(times)-1):
        #print(i)
        if i%every==0:
            sys.stdout.write("%.0f Percent done"%(10*(i/every))+"."*10+"%.8f\n"%(time()-btime))
            sys.stdout.flush()
        # compute any expectation values
        if i%1==0:
            pops = wf.diabatic_pops()
            f.write('%.8f '%(times[i]))
            for j in range(len(pops)):
                f.write('%.8f '%(pops[j]))
        if i==0:
            opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        mfs,rhos,projs = cmfprecomputemf(ham,opspfs,opips,spfovs,wf)
        # integrate coefficients one half timestep forward
        #print('A half')
        Atmp = wf.copy('A')
        Aintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, opips, spfovs)
        # integrate spfs one half timestep forward
        #print('spfs ghost half')
        spfstmp = wf.copy('spfs')
        spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs, 
                      cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
                      spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        # precompute new stuff for propagation
        opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        mfs,rhos,projs = cmfprecomputemf(ham,opspfs,opips,spfovs,wf)
        # integrate spfs one full timestep forward
        wf.spfs = deepcopy(spfstmp)
        #print('spfs full')
        #spfsintegrate(times[i], times[i+1], dt, wf, ham, pbfs, 
        #              cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
        #              spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs, 
                      cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
                      spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        spfsintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, pbfs, 
                      cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
                      spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        #precompute new stuff for propagation
        opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        # compute error in cmf propagation
        if comperror:
            Atmp2 = wf.copy('A')
            Aintegrate(times[i]+0.5*dt, times[i], -0.5*dt, wf, ham, opips, spfovs)
            diff = 0.0
            for j in range(nel):
                diff += np.linalg.norm(Atmp[j]-wf.A[j])
            tot = 0.0
            for j in range(nel):
                tot += np.sum(np.abs(wf.A[j]))
            ferr.write('%.16f %.16f %.16f %.16f\n'%(times[i],diff,tot,diff/tot))
            ferr.flush()
            # integrate coefficients one half timestep forward
            wf.A = deepcopy(Atmp2)
        # integrate coefficients one half timestep forward
        #print('A full')
        Aintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, opips, spfovs)
        if i%1==0:
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
    f.close()
    if comperror:
        ferr.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    return wf

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

    # set up integrator and options
    dt = times[1]-times[0]
    Aintegrate = integrator.krylov_prop
    spfsintegrate = integrator.rk

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
        if i==0:
            uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        mfs,rhos,projs = cmfprecomputemf(ham,copips,spfovs,wf)
        # integrate coefficients one half timestep forward
        Atmp = wf.copy('A')
        Aintegrate(times[i],times[i]+0.5*dt,0.5*dt,wf,ham,uopips,copips,spfovs)
        # integrate spfs one half timestep forward
        spfstmp = wf.copy('spfs')
        spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs,
                      cmfflag=True, eq='spfs', opspfs=[uopspfs,copspfs],
                      opips=[uopips,copips], spfovs=spfovs, mfs=mfs, rhos=rhos,
                      projs=projs)
        # precompute new stuff for propagation
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        mfs,rhos,projs = cmfprecomputemf(ham,copips,spfovs,wf)
        # integrate spfs one half timestep forward
        wf.spfs = deepcopy(spfstmp)
        #spfsintegrate(times[i], times[i+1], dt, wf, ham, pbfs,
        #              cmfflag=True, eq='spfs', opspfs=[uopspfs,copspfs],
        #              opips=[uopips,copips], spfovs=spfovs, mfs=mfs, rhos=rhos,
        #              projs=projs)
        spfsintegrate(times[i], times[i+1]+0.5*dt, 0.5*dt, wf, ham, pbfs,
                      cmfflag=True, eq='spfs', opspfs=[uopspfs,copspfs],
                      opips=[uopips,copips], spfovs=spfovs, mfs=mfs, rhos=rhos,
                      projs=projs)
        spfsintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, pbfs,
                      cmfflag=True, eq='spfs', opspfs=[uopspfs,copspfs],
                      opips=[uopips,copips], spfovs=spfovs, mfs=mfs, rhos=rhos,
                      projs=projs)
        #precompute new stuff for propagation
        uopspfs,copspfs,uopips,copips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        # integrate coefficients one half timestep forward
        Aintegrate(times[i]+0.5*dt,times[i],0.5*dt,wf,ham,uopips,copips,spfovs)
        if i%1==0:
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
    f.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    return wf

if __name__ == "__main__":

    import numpy as np
    np.set_printoptions(precision=20)
    import units
    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from pbasis import PBasis
    from optools import precompute_ops

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

    dt = 0.5
    times = np.arange(0.0,120.,dt)*units.convert_to('fs')

    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4.txt')
    #wf = vmfpropagate(times, wf, ham, 'nonhermitian.txt')
