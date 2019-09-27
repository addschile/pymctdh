import numpy as np
import integrator
from eom import eom
from time import time
import sys

def vmfpropagate(times, ham, pbfs, wf, filename):#, options=None, results=None):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """
    # set up integrator and options
    dt = times[1]-times[0]
    integrate = integrator.rk

    # compute initial values to keep track
    normprev = wf.norm()

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
        # integrate one timestep forward
        integrate(times[i], times[i+1], dt, wf, ham, pbfs)
        #energy,error = integrate(times[i], times[i+1], dt, wf, ham, pbfs)
        if i%1==0:
            norm = wf.norm()
            f.write('%.8f\n'%(norm))
            #f.write('%.8f %.8f %.8f\n'%(energy,norm,error))
            f.flush()
    f.close()
    sys.stdout.write("100 Percent done"+"."*10+"%.8f\n"%(time()-btime))
    sys.stdout.flush()
    return wf

def vmfpropagatejumps(times, wf, ham, Ls, LdLs, filename, seed=None, 
                      nguess=1000, jump_time_tol=1.e-3):#, options=None, results=None):
    """Propagate MCTDH wavefunction based on Dirac-Frenkel variational
    principle. Uses the variable mean field scheme in which 

    Inputs
    ------
    Outputs
    -------
    """
    from optools import jump

    # set up random number generator
    if seed==None:
        seeder = int(time())
        rng = np.random.RandomState(seed=seeder)
    else:
        seeder = seed
        rng = np.random.RandomState(seed=seeder)

    # set up integrator and options
    dt = times[1]-times[0]
    integrate = integrator.rk

    f = open(filename,'w')

    wf_track = wf.copy()
    rand = rng.uniform()
    njumps = 0
    jumps = []
    for i in range(len(times)-1):
        # compute any expectation values
        if i%1==0:
            pops = wf_track.diabatic_pops()
            f.write('%.8f '%(times[i]))
            for j in range(len(pops)):
                f.write('%.8f '%(pops[j]))
            norm = wf_track.norm()
            f.write('%.8f\n'%(norm))
            f.flush()
        tau = times[i]
        while tau != times[i+1]:
            # data before integrating
            t_prev = tau
            wf_prev = wf.copy()
            norm_prev = wf.norm()

            # integrate one timestep forward
            wf,energy,error = integrate(t_prev, times[i+1], times[i+1]-t_prev, wf, ham)

            # compute new norm
            norm_psi = wf.norm()
            t_next = times[i+1]
            #print(t_next,norm_psi,rand)

            if norm_psi <= rand:

                # quantum jump has happened
                njumps += 1

                ii = 0
                t_final = t_next

                while ii < nguess:

                    ii += 1

                    t_guess = t_prev + np.log(norm_prev / rand) / \
                        np.log(norm_prev / norm_psi)*(t_final-t_prev)

                    # integrate psi from t_prev to t_guess
                    norm_prev = wf_prev.norm()
                    wf_guess,energy,error = integrate(t_prev, t_guess, (t_guess-t_prev), wf_prev, ham)
                    norm_guess = wf_guess.norm()
                    #print(t_guess,norm_psi,norm_prev,norm_guess,norm_guess-rand,t_guess-t_prev)

                    # determine what to do next
                    if (np.abs(norm_guess - rand) <= (jump_time_tol*rand)):
                        # t_guess was right!
                        tau = t_guess

                        # jump
                        wf_guess.normalize()
                        rand = rng.uniform()
                        wf , ind = jump(rand, Ls, LdLs, wf_guess)
                        jumps.append( [tau,ind] )

                        # choose a new random number for next jump
                        rand = rng.uniform()
                        break
                    elif (norm_guess < rand):
                        # t_guess > t_jump
                        t_final = t_guess
                        norm_psi = norm_guess
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        wf_prev = wf_guess.copy()
                        norm_prev = norm_guess
                    if ii == nguess:
                        raise ValueError("Couldn't find jump time")
            else:
                # no jump update time
                tau = times[i+1]
                # store new normalized wavefunction for this timestep
                wf_track = wf.copy()
                wf_track.normalize()

    f.close()

    return wf

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
