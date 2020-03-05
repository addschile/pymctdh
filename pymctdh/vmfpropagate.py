import sys
from time import time

import numpy as np
import scipy.integrate

from pymctdh.eom import vmfeom
from pymctdh.results import Results
from pymctdh.cy.wftools import normalize_wf

def vmfpropagate(times, ham, pbfs, wf, results=None, return_wf=False):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """

    # set up results
    if results is None:
        results = Results()

    # set up integrator
    dt = times[1]-times[0]
    r  = scipy.integrate.ode(vmfeom)
    r.set_integrator('zvode', method='adams', order=12, atol=1.e-8, rtol=1.e-6,
                     nsteps=1000, first_step=0, min_step=0, max_step=dt)
    ode_args  = [wf.npsi,wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.psistart,wf.psiend,
                 wf.spfstart,wf.spfend,ham,pbfs]
    r.set_f_params(*ode_args)
    r.set_initial_value(wf.psi, times[0])

    if len(times) >= 10:
        every = int(len(times)/10)
    else:
        every = 1

    # propagate wavefunction
    btime = time()
    for i in range(len(times)):

        # print status update
        if i%every==0:
            sys.stdout.write("%.0f Percent done"%(10*(i/every))+"."*10+"%.8f\n"%(time()-btime))
            sys.stdout.flush()

        # analyze wavefunction
        results.analyze_state(i, times[i], r.y, wf.info)

        # integrate one timestep forward
        r.integrate(r.t + dt)

    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()

    results.close_down()

    if return_wf:
        wf.psi = r.y.copy()
        return wf
    else:
        return 

def vmfpropagatejumps(times, ham, pbfs, Ls, LdLs, wf, ntraj=100, results=None, seed=None,
                      nguess=1000, jump_time_tol=1.e-3):
    """
    """

    # set up results
    if results is None:
        results = Results()

    # set up random number generator
    if seed==None:
        seeder = int(time())
        rng = np.random.RandomState(seed=seeder)
    else:
        seeder = seed
        rng = np.random.RandomState(seed=seeder)

    # set up integrator and options
    dt = times[1]-times[0]
    r  = scipy.integrate.ode(vmfeom)
    r.set_integrator('zvode', method='adams', order=12, atol=1.e-8, rtol=1.e-6,
                     nsteps=1000, first_step=0, min_step=0, max_step=dt)
    ode_args  = [wf.npsi,wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.psistart,wf.psiend,
                 wf.spfstart,wf.spfend,ham,pbfs]
    r.set_f_params(*ode_args)

    if ntraj >= 10:
        every = int(float(ntraj)/10)
    else:
        every = 1

    # propagate trajectories
    btime = time()
    for traj in range(ntraj):

        # print status update
        if traj%every==0:
            sys.stdout.write("%.0f Percent done"%(10*(traj/every))+"."*10+"%.8f\n"%(time()-btime))
            sys.stdout.flush()

        # make new results class for this trajectory
        results_traj = results.copy(traj=traj)

        # set initial condition
        r.set_initial_value(wf.psi.copy(), times[0])

        # propagate one trajectory
        vmfpropjumpstraj(times,ham,pbfs,Ls,LdLs,wf,r,rng,results=results_traj,
                         nguess=1000,jump_time_tol=1.e-3)

        # add new results to previous results
        results.add(results_traj)

    # average results at the end of the run
    results.average()

    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()

    return

def vmfpropjumpstraj(times, ham, pbfs, Ls, LdLs, wf, r, rng, results=None,
                      nguess=1000, jump_time_tol=1.e-3):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """
    psi_track = r.y.copy()
    rand = rng.uniform()
    njumps = 0
    jumps = []
    for i in range(len(times)-1):

        # analyze wavefunction
        results.analyze_state(i, times[i], psi_track, wf.info)

        tau = times[i]
        while tau != times[i+1]:

            # data before integrating
            t_prev = tau
            psi_prev = r.y.copy()
            ind = wf.psiend[0,-1]
            norm_prev = np.sum((r.y[:ind].conj()*r.y[:ind])).real

            # integrate one timestep forward
            r.integrate(r.t + (times[i+1]-tau))

            # compute new norm
            ind = wf.psiend[0,-1]
            norm_psi = np.sum((r.y[:ind].conj()*r.y[:ind])).real
            t_next = times[i+1]

            if norm_psi <= rand:

                r.set_initial_value(psi_prev, tau)

                # quantum jump has happened
                njumps += 1

                ii = 0
                t_final = t_next

                while ii < nguess:

                    ii += 1

                    t_guess = t_prev + np.log(norm_prev / rand) / \
                        np.log(norm_prev / norm_psi)*(t_final-t_prev)

                    # integrate psi from t_prev to t_guess
                    r.integrate(r.t + (t_guess-t_prev))
                    ind = wf.psiend[0,-1]
                    norm_guess = np.sum((r.y[:ind].conj()*r.y[:ind])).real

                    # determine what to do next
                    if (np.abs(norm_guess - rand) <= (jump_time_tol*rand)):
                        # t_guess was right!
                        tau = t_guess

                        # reset wavefunction
                        normalize_wf(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.psistart,wf.psiend,r.y)

                        # jump
                        rand = rng.uniform()
                        ind = jump(rand, Ls, LdLs, wf, pbfs, r.y)
                        jumps.append( [tau,ind] )
                        normalize_wf(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.psistart,wf.psiend,r.y)

                        # reset integrator
                        r.set_initial_value(r.y, tau)

                        # choose a new random number for next jump
                        rand = rng.uniform()
                        break
                    elif (norm_guess < rand):
                        # t_guess > t_jump
                        t_final  = t_guess
                        norm_psi = norm_guess
                        r.set_initial_value(psi_prev, t_prev)
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        norm_prev = norm_guess
                    if ii == nguess:
                        raise ValueError("Couldn't find jump time")
            else:
                # no jump update time
                tau = times[i+1]
                # store new normalized wavefunction for this timestep
                psi_track = r.y.copy()
                normalize_wf(wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.spfstart,wf.psistart,wf.psiend,psi_track)
    return

def jump_probs(LdLs,wf,pbfs,psi):
    """Computes the jump probabilities of all the quantum jump operators.
    """
    from pymctdh.expect import compute_expect
    # get wf info
    nmodes   = wf.nmodes
    nel      = wf.nel
    nspfs    = wf.nspfs
    npbfs    = wf.npbfs
    spfstart = wf.spfstart
    spfend   = wf.spfend
    psistart = wf.psistart
    psiend   = wf.psiend
    # compute expects
    p_n = np.zeros(len(LdLs))
    for i in range(len(LdLs)):
        p_n[i] = compute_expect(nel,nmodes,nspfs,npbfs,spfstart,spfend,psistart,
                                psiend,psi,LdLs[i],pbfs)
    p = np.sum(p_n)
    return p_n , p

def jump(rand,Ls,LdLs,wf,pbfs,psi):
    """
    """
    from pymctdh.optools import act_operator
    # reshpae y into A tensor and spfs
    A = np.zeros(2, dtype=np.ndarray)
    spfs = np.zeros(2, dtype=np.ndarray)
    for alpha in range(wf.nel):
        shaper = ()
        for mode in range(wf.nmodes):
            shaper += (wf.nspfs[alpha,mode],)
        # set A
        ind0 = wf.psistart[0,alpha]
        indf = wf.psiend[0,alpha]
        A[alpha] = np.reshape(psi[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = wf.psistart[1,alpha]
        indf = wf.psiend[1,alpha]
        spfs[alpha] = psi[ind0:indf]

    # compute jump probabilities
    p_n , p = jump_probs(LdLs,wf,pbfs,psi)
    p_n = np.cumsum(p_n)
    # see which one it jumped along
    p *= rand
    for count in range(len(Ls)):
        if p <= p_n[count]:
            act_operator(A,spfs,wf,Ls[count],pbfs,psi)
            return count
