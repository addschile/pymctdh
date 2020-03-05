import numpy as np
import pymctdh

if __name__ == "__main__":

    nel    = 2
    nmodes = 2
    nspfs = np.array([[10,10],
                      [8,8]], dtype=int)
    npbfs = [[18, 28],[18, 14]]

    pbfs = list()
    mode1 = [['ho', npbfs[0][0], 1.0, 1.0],['ho', npbfs[0][1], 1.0, 1.0]]
    mode2 = [['ho', npbfs[1][0], 1.0, 1.0],['ho', npbfs[1][1], 1.0, 1.0]]
    pbfs.append( pymctdh.PBasis(mode1, combined=True, sparse=True) )
    pbfs.append( pymctdh.PBasis(mode2, combined=True, sparse=True) )

    wf = pymctdh.Wavefunction(nel, nmodes, nspfs, npbfs)
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
    # combined mode 0
    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 0, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 0, 'ops': '(1)*(q^2)'})
    # combined mode 1
    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 1, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 1, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 1, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 1, 'ops': '(1)*(q^2)'})
    # combined mode 0
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': '(q)*(1)'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 1
    # combined mode 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 1

    ham = pymctdh.Hamiltonian(nel, nmodes, hterms, pbfs=pbfs)

    dt = 0.5
    times = np.arange(0.0,120.,dt)*pymctdh.units.convert_to('fs')

    results = pymctdh.Results(nsteps=len(times), db_pops=True, print_db_pops=True, db_pops_file='pyr4_combined_sparse.txt')
    #results = pymctdh.Results(nsteps=len(times), db_pops=True, print_db_pops=True, db_pops_file='pyr4_combined_sparse.txt',
    #                          store_states=True, store_every=10, print_restart=True, restart_file='psi_restart', restart_every=100,
    #                          print_states=True, states_file='wf/psi', states_every=10)
    output = pymctdh.vmfpropagate(times, ham, pbfs, wf, results=results)
    #print(len(output.states))
