import numpy as np
import integrator
from optools import precompute_ops
from cy.wftools import (overlap_matrices,compute_density_matrix,
                        invert_density_matrix,compute_projector)
from meanfield import compute_meanfield_mats
from copy import deepcopy
from time import time
import sys

#def cmfprecompute(ham, pbfs, wf):
#    """
#    """
#    # get wavefunction info
#    nel      = wf.nel
#    nmodes   = wf.nmodes
#    nspfs    = wf.nspfs
#    npbfs    = wf.npbfs
#    spfstart = wf.spfstart
#    spfend   = wf.spfend
#
#     precompute stuff for propagation
#    opspfs, opips = precompute_ops(nel,nmodes,nspfs,npbfs,spfstart,spfend,
#                                   ham.ops,pbfs,wf.spfs)
#    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,wf.spfs)
#
#    mfs = compute_meanfield_mats(nel,nmodes,nspfs,npbfs,spfstart,
#                                     spfend,ham.hcterms,opspfs,opips,
#                                     spfovs,wf.A,wf.spfs)
#    rhos  = []
#    projs = []
#    for alpha in range(nel):
#        rho_tmp = []
#        proj_tmp = []
#        for mode in range(nmodes):
#            nspf = nspfs[alpha,mode]
#            npbf = npbfs[mode]
#            ind0 = spfstart[alpha,mode]
#            indf = spfend[alpha,mode]
#            # compute and invert density matrix
#            rho = compute_density_matrix(nspf,alpha,nmodes,mode,wf.A[alpha])
#            rho_tmp.append( invert_density_matrix(rho) )
#            # compute projector
#            proj_tmp.append( compute_projector(nspf,npbf,wf.spfs[alpha][ind0:indf]) )
#        rhos.append( rho_tmp )
#        projs.append( proj_tmp )
#    return opspfs,opips,spfovs,mfs,rhos,projs

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
    opspfs, opips = precompute_ops(nel,nmodes,nspfs,npbfs,spfstart,spfend,
                                   ham.ops,pbfs,wf.spfs)
    spfovs = overlap_matrices(nel,nmodes,nspfs,npbfs,spfstart,wf.spfs)
    return opspfs,opips,spfovs

def cmfprecomputemf(ham, opspfs, opips, spfovs, wf):
    """
    """
    # get wavefunction info
    nel      = wf.nel
    nmodes   = wf.nmodes
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend

    mfs = compute_meanfield_mats(nel,nmodes,nspfs,npbfs,spfstart,
                                     spfend,ham.hcterms,opspfs,opips,
                                     spfovs,wf.A,wf.spfs)
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

def cmfpropagate(times, ham, pbfs, wf, filename):
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
        print('tstep',i)
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
        Atmp = wf.copy('A')
        Aintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, opips, spfovs)
        # integrate spfs one half timestep forward
        spfstmp = wf.copy('spfs')
        spfsintegrate(times[i], times[i]+0.5*dt, 0.5*dt, wf, ham, pbfs, 
                      cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
                      spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        # precompute new stuff for propagation
        opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        mfs,rhos,projs = cmfprecomputemf(ham,opspfs,opips,spfovs,wf)
        # integrate spfs one half timestep forward
        wf.spfs = deepcopy(spfstmp)
        spfsintegrate(times[i], times[i+1], dt, wf, ham, pbfs, 
                      cmfflag=True, eq='spfs', opspfs=opspfs, opips=opips, 
                      spfovs=spfovs, mfs=mfs, rhos=rhos, projs=projs)
        #precompute new stuff for propagation
        opspfs,opips,spfovs = cmfprecomputemels(ham,pbfs,wf)
        # integrate coefficients one half timestep forward
        Atmp2 = wf.copy('A')
        Aintegrate(times[i]+0.5*dt, times[i], -0.5*dt, wf, ham, opips, spfovs)
        diff = 0.0
        for j in range(nel):
            diff += np.linalg.norm(Atmp[j]-wf.A[j])
        tot = 0.0
        for j in range(nel):
            tot += np.sum(np.abs(wf.A[j]))
        print(diff,diff/tot)
        wf.A = deepcopy(Atmp2)
        Aintegrate(times[i]+0.5*dt, times[i+1], 0.5*dt, wf, ham, opips, spfovs)
        if i%1==0:
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
    f.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    return wf

#def vmfpropagatejumps(times, ham, pbfs, Ls, LdLs, wf, filename, seed=None, 
#                      nguess=1000, jump_time_tol=1.e-3):#, options=None, results=None):
#    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
#    principle. Uses the variable mean field scheme in which 
#
#    Inputs
#    ------
#    Outputs
#    -------
#    """
#    from optools import jump
#
#    # set up random number generator
#    if seed==None:
#        seeder = int(time())
#        rng = np.random.RandomState(seed=seeder)
#    else:
#        seeder = seed
#        rng = np.random.RandomState(seed=seeder)
#
#    # set up integrator and options
#    dt = times[1]-times[0]
#    integrate = integrator.rk
#
#    f = open(filename,'w')
#
#    btime = time()
#    wf_track = wf.copy()
#    rand = rng.uniform()
#    njumps = 0
#    jumps = []
#    for i in range(len(times)-1):
#        # compute any expectation values
#        if i%1==0:
#            pops = wf_track.diabatic_pops()
#            f.write('%.8f '%(times[i]))
#            for j in range(len(pops)):
#                f.write('%.8f '%(pops[j]))
#            norm = wf_track.norm()
#            f.write('%.8f\n'%(norm))
#            f.flush()
#        tau = times[i]
#        while tau != times[i+1]:
#            # data before integrating
#            t_prev = tau
#            wf_prev = wf.copy()
#            norm_prev = wf.norm()
#
#            # integrate one timestep forward
#            integrate(t_prev, times[i+1], times[i+1]-t_prev, wf, ham, pbfs)
#            #wf,energy,error = integrate(t_prev, times[i+1], times[i+1]-t_prev, wf, ham)
#
#            # compute new norm
#            norm_psi = wf.norm()
#            t_next = times[i+1]
#            #print(t_next,norm_psi,rand)
#
#            if norm_psi <= rand:
#
#                wf_guess = wf_prev.copy()
#
#                # quantum jump has happened
#                njumps += 1
#
#                ii = 0
#                t_final = t_next
#
#                while ii < nguess:
#
#                    ii += 1
#
#                    t_guess = t_prev + np.log(norm_prev / rand) / \
#                        np.log(norm_prev / norm_psi)*(t_final-t_prev)
#
#                    # integrate psi from t_prev to t_guess
#                    norm_prev = wf_prev.norm()
#                    integrate(t_prev, t_guess, (t_guess-t_prev), wf_guess, ham, pbfs)
#                    #wf_guess,energy,error = integrate(t_prev, t_guess, (t_guess-t_prev), wf_prev, ham)
#                    norm_guess = wf_guess.norm()
#                    #print(t_guess,norm_psi,norm_prev,norm_guess,norm_guess-rand,t_guess-t_prev)
#
#                    # determine what to do next
#                    if (np.abs(norm_guess - rand) <= (jump_time_tol*rand)):
#                        # t_guess was right!
#                        tau = t_guess
#
#                        # jump
#                        wf_guess.normalize()
#                        rand = rng.uniform()
#                        ind = jump(rand, Ls, LdLs, wf_guess, pbfs)
#                        jumps.append( [tau,ind] )
#                        wf = wf_guess.copy()
#
#                        # choose a new random number for next jump
#                        rand = rng.uniform()
#                        break
#                    elif (norm_guess < rand):
#                        # t_guess > t_jump
#                        t_final = t_guess
#                        wf_guess = wf_prev.copy()
#                        norm_psi = norm_guess
#                    else:
#                        # t_guess < t_jump
#                        t_prev = t_guess
#                        #wf_prev = wf_guess.copy()
#                        norm_prev = norm_guess
#                    if ii == nguess:
#                        raise ValueError("Couldn't find jump time")
#            else:
#                # no jump update time
#                tau = times[i+1]
#                # store new normalized wavefunction for this timestep
#                wf_track = wf.copy()
#                wf_track.normalize()
#
#    f.close()
#    print(time()-btime)
#
#    return wf

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
    #hterms.append({'coeff':   -0.1*0.5j, 'units': 'ev', 'modes': 0, 'elop':   '1', 'ops': 'q^2'}) # parameters for effective hamiltonian
    #hterms.append({'coeff':   -0.1*0.5j, 'units': 'ev', 'modes': 1, 'elop':   '1', 'ops': 'q^2'}) # parameters for effective hamiltonian
    #hterms.append({'coeff':   -0.1*0.5j, 'units': 'ev', 'modes': 2, 'elop':   '1', 'ops': 'q^2'}) # parameters for effective hamiltonian
    #hterms.append({'coeff':   -0.1*0.5j, 'units': 'ev', 'modes': 3, 'elop':   '1', 'ops': 'q^2'}) # parameters for effective hamiltonian

    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    dt = 0.5
    times = np.arange(0.0,120.,dt)*units.convert_to('fs')

    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4.txt')
    #wf = vmfpropagate(times, wf, ham, 'nonhermitian.txt')
