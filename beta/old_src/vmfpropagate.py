import sys
from time import time

import numpy as np
import scipy.integrate

from eom import vmfeom

def vmfpropagate(times, ham, pbfs, wf, filename):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """
    # set up integrator and options
    dt = times[1]-times[0]
    r  = scipy.integrate.ode(vmfeom)
    r.set_integrator('zvode', method='adams', order=12, atol=1.e-8, rtol=1.e-6,
                     nsteps=1000, first_step=0, min_step=0, max_step=dt)
    ode_args  = [wf.npsi,wf.nel,wf.nmodes,wf.nspfs,wf.npbfs,wf.psistart,wf.psiend,
                 wf.spfstart,wf.spfend,ham,pbfs]
    r.set_f_params(*ode_args)
    r.set_initial_value(wf.psi, times[0])

    f = open(filename,'w')
    every = int(len(times)/10)
    btime = time()
    for i in range(len(times)-1):
        if i%every==0:
            sys.stdout.write("%.0f Percent done"%(10*(i/every))+"."*10+"%.8f\n"%(time()-btime))
            sys.stdout.flush()
        # compute any expectation values
        if i%1==0:
            wf.psi = r.y
            pops = wf.diabatic_pops()
            f.write('%.8f '%(times[i]))
            for j in range(len(pops)):
                f.write('%.8f '%(pops[j]))
        # integrate one timestep forward
        r.integrate(r.t + dt)#, relax=True)
        if i%1==0:
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
    f.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    wf.psi = r.y.copy()
    return wf

def vmfpropagatejumps(times, ham, pbfs, Ls, LdLs, wf, filename, seed=None, 
                      nguess=1000, jump_time_tol=1.e-3):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """

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
    r.set_initial_value(wf.psi, times[0])

    f = open(filename,'w')
    btime = time()
    rand = rng.uniform()
    njumps = 0
    jumps = []
    for i in range(len(times)-1):
        # compute any expectation values
        if i%1==0:
            pops = wf.diabatic_pops()
            f.write('%.8f '%(times[i]))
            for j in range(len(pops)):
                f.write('%.8f '%(pops[j]))
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
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

            #print(norm_psi,rand)
            # determine what to do next
            #if (np.abs(norm_psi - rand) <= (jump_time_tol*rand)):
            #    # t_guess was right!
            #    tau = times[i+1]

            #    # reset wavefunction
            #    wf.psi = r.y.copy()
            #    wf.normalize()

            #    # jump
            #    rand = rng.uniform()
            #    ind = jump(rand, Ls, LdLs, wf, pbfs)
            #    jumps.append( [tau,ind] )

            #    # reset integrator
            #    r.set_initial_value(wf.psi, tau)

            #    # choose a new random number for next jump
            #    rand = rng.uniform()
            if norm_psi <= rand:
                print('hey',times[i],times[i+1],norm_prev,norm_psi,rand)

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

                    print(t_prev,t_guess,t_final,norm_prev,norm_guess,rand)

                    # determine what to do next
                    if (np.abs(norm_guess - rand) <= (jump_time_tol*rand)):
                        # t_guess was right!
                        tau = t_guess

                        # reset wavefunction
                        wf.psi = r.y.copy()
                        wf.normalize()

                        # jump
                        rand = rng.uniform()
                        ind = jump(rand, Ls, LdLs, wf, pbfs)
                        jumps.append( [tau,ind] )
                        wf.normalize()

                        # reset integrator
                        r.set_initial_value(wf.psi, tau)

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
                wf.psi = r.y.copy()
                wf.normalize()

    f.close()
    print(time()-btime)

    return wf

def jump_probs(LdLs,wf,pbfs):
    """Computes the jump probabilities of all the quantum jump operators.
    """
    from expect import compute_expect
    # compute expects
    p_n = np.zeros(len(LdLs))
    for i in range(len(LdLs)):
        p_n[i] = compute_expect(LdLs[i],wf,pbfs)
    p = np.sum(p_n)
    return p_n , p

def jump(rand,Ls,LdLs,wf,pbfs):
    """
    """
    from optools import act_operator
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
        A[alpha] = np.reshape(wf.psi[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = wf.psistart[1,alpha]
        indf = wf.psiend[1,alpha]
        spfs[alpha] = wf.psi[ind0:indf]

    # compute jump probabilities
    p_n , p = jump_probs(LdLs,wf,pbfs)
    p_n = np.cumsum(p_n)
    # see which one it jumped along
    p *= rand
    for count in range(len(Ls)):
        if p <= p_n[count]:
            act_operator(A,spfs,wf,Ls[count],pbfs)
            return count

if __name__ == "__main__":

    import numpy as np
    np.set_printoptions(precision=20)
    import units
    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from pbasis import PBasis
    from optools import precompute_ops
    from eom import eom_coeffs

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
    gam   =  0.0050

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
    #hterms.append({'coeff': -0.5j*gam, 'units': 'ev', 'modes': 0, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -0.5j*gam, 'units': 'ev', 'modes': 1, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -0.5j*gam, 'units': 'ev', 'modes': 2, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian
    #hterms.append({'coeff': -0.5j*gam, 'units': 'ev', 'modes': 3, 'elop':   '1', 'ops': 'n'}) # parameters for effective hamiltonian

    ham = Hamiltonian(nmodes, hterms, pbfs=pbfs)

    dt = 0.5
    times = np.arange(0.0,120.,dt)*units.convert_to('fs')

    wf = vmfpropagate(times, ham, pbfs, wf, 'pyr4.txt')
    #wf = vmfpropagate(times, wf, ham, 'nonhermitian.txt')
