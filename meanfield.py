import numpy as np
from optools import compute_el_mes
from cy.wftools import spf_innerprod,multAtens
from tensorutils import atensorcontract,ahtensorcontract

def act_meanfield(alpha, beta, mode, mf, spfs):
    """Computes the action of the meanfield operators on the spfs from stored
    meanfield operators. This is only used in the CMF integration scheme.
    """
    # TODO
    # this needs to be tensorialized because spfs will be too big
    # mf is nspf x nspf
    # spfs is (nspf*npbf) x 1
    return np.dot(mf[alpha][beta][mode], spfs)

def compute_meanfield_uncorr(alpha, mode, hterms, opspfs, spfs, storemf=False, 
                             gen=False):
    """Computes the meanfield operators for uncorrelated hamiltonian terms.
    Input
    -----
    alpha - int, the bra electronic state
    mode  - int, the mode being acted on
    hterms - list of dictionaries, the correlated hamiltonian terms.
    opspfs - list, the list/dictionary structure that stores the hamiltonian
             terms acting on each spf
    spfs - Wavefunction class, the wavefunction for which mf ops are being 
         computed and acted
    storemf - bool, determines whether or not to store meanfield operators,
              used in cmf integration scheme (not implemented tho)
    gen - bool,

    Output
    ------
    spfout - np.ndarray, output of meanfield operators acting on input spf
    mfops - (not implemented) meanfield operators, only returned in cmf scheme
    """
    # get wf info
    spfout = np.zeros_like(spfs, dtype=complex)
    for hterm in hterms:
        # compute electronic matrix element
        hel = compute_el_mes(alpha,alpha,hterm['elop'])
        if hel != 0.0:
            hel *= hterm['coeff']
            if 'modes' in hterm:
                if mode == hterm['modes'][0]:
                    spfout += hel*opspfs[alpha][mode][hterm['ops'][0]]
                else:
                    if gen:
                        spfout += spfs
            else:
                spfout += hel*spfs
    return spfout

def compute_meanfield_corr(nmodes, nspfs, npbfs, spfstart, spfend, alpha, beta, 
                           hterms, opspfs, opips, spfovs, A, spfs, storemf=False):
    """Computes the meanfield operators and their action on the spfs.
    Meanfield operators are stored if using the CMF integrator.

    Input
    -----
    alpha - int, the bra electronic state
    beta  - int, the ket electronic state
    wf - Wavefunction class, the wavefunction for which mf ops are being 
         computed and acted
    ham - Hamiltonian class, the hamiltonian of the system
    opspfs - list, the list/dictionary structure that stores the hamiltonian
             terms acting on each spf
    opips - list, the list/dictionary structure that stores the inner products
            of the hamiltonian terms for each spf
    storemf - bool, determines whether or not to store meanfield operators,
              used in cmf integration scheme (not implemented tho)

    Output
    ------
    spfout - np.ndarray, output of meanfield operators acting on input spf
    mfops - (not implemented) meanfield operators, only returned in cmf scheme

    Notes
    -----
    For each term in the Hamiltonian the order of operators for how the
    mean field operator acts on the spfs is:
    1) contract the A tensor over all modes that aren't acted on by the term
    2) compute the inner products of all the spfs that are acted on by the term
    3) contract over the contracted A tensor and the matrix elements of the modes
    4) act the meanfield operator on the vector of spfs
    """

    spfout = np.zeros_like(spfs[alpha], dtype=complex)
    for mode in range(nmodes):
        ind0    = spfstart[alpha,mode]
        indf    = spfend[alpha,mode]
        nspf_l  = nspfs[alpha,mode]
        nspf_r  = nspfs[beta,mode]
        npbf    = npbfs[mode]
        spfout_ = spfout[ind0:indf]
        for hterm in hterms:
            # compute electronic matrix element
            hel = compute_el_mes(alpha,beta,hterm['elop'])
            if hel != 0.0:
                hel *= hterm['coeff']
                if 'modes' in hterm:
                    hmodes = hterm['modes'].copy()
                    if mode not in hmodes:
                        hmodes += [mode]
                    # contract over all other modes not involved in the term
                    if alpha == beta:
                        Asum = atensorcontract(hmodes, A[beta])
                    elif alpha < beta:
                        Asum = atensorcontract(hmodes, A[alpha], A[beta], 
                                               spfovs=spfovs[alpha][beta-alpha-1])
                    else:
                        Asum = atensorcontract(hmodes, A[alpha], A[beta], 
                                               spfovs=spfovs[beta][alpha-beta-1],
                                               spfovsconj=True)
                    # contract over inner products of ham terms of other modes
                    for num,hmode in enumerate(hterm['modes']):
                        if not hmode == mode:
                            op = hterm['ops'][num]
                            if alpha <= beta:
                                opsum = opips[alpha][beta-alpha][hmode][op]
                                conj = False
                            else:
                                opsum = opips[beta][alpha-beta][hmode][op].conj()
                                conj = True
                            if hmode < mode:
                                order = 0
                            else:
                                order = 1
                            Asum = ahtensorcontract(Asum,opsum,order,conj=conj)
                        elif hmode == mode:
                            modeop = hterm['ops'][num]
                    if mode in hterm['modes']:
                        spfout_ += hel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                 opspfs[beta][mode][modeop])
                    else:
                        ind0 = spfstart[beta,mode]
                        indf = spfend[beta,mode]
                        spfout_ += hel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                 spfs[beta][ind0:indf])
                else: # purely electronic operator
                    # contract over all other modes
                    if alpha == beta:
                        Asum = atensorcontract([mode], A[beta])
                    elif alpha < beta:
                        Asum = atensorcontract([mode], A[alpha], A[beta],
                                               spfovs=spfovs[alpha][beta-alpha-1])
                    else:
                        Asum = atensorcontract([mode], A[alpha], A[beta],
                                               spfovs=spfovs[beta][alpha-beta-1],
                                               spfovsconj=True)
                    ind0 = spfstart[beta,mode]
                    indf = spfend[beta,mode]
                    spfout_ += hel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                             spfs[beta][ind0:indf])
    return spfout
    # TODO how to store meanfield operators
    #if storemf:
    #    return spfout , mf ops
    #else:
    #    return spfout

if __name__ == "__main__":

    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from optools import precompute_ops
    from time import time

    nel    = 2
    nmodes = 4
    nspf = np.array([[8, 13, 7, 6],
                     [7, 12, 6, 5]], dtype=int)
    pbf = list()
    pbf.append( ['ho', 22, 1.0, 1.0] )
    pbf.append( ['ho', 32, 1.0, 1.0] )
    pbf.append( ['ho', 21, 1.0, 1.0] )
    pbf.append( ['ho', 12, 1.0, 1.0] )

    wf = Wavefunction(nel, nmodes, nspf, pbf)
    wf.generate_ic(1)

    w10a  = 0.09357
    w6a   = 0.0740
    w1    = 0.1273
    w9a   = 0.1568
    delta = 0.46165
    lamda = 0.1825
    k6a1  =-0.0964
    k6a2  = 0.1194
    k11   = 0.0470
    k12   = 0.2012
    k9a1  = 0.1594
    k9a2  = 0.0484

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

    ham = Hamiltonian(nmodes, hterms)

    wf.overlap_matrices()
    opspfs,opips = precompute_ops(ham.ops, wf)

    btime = time()
    for i in range(int(1e3)):
        for alpha in range(nel):
            for beta in range(nel):
                spfout = compute_meanfield_corr(alpha,beta,wf,ham.hcterms,opspfs,opips)
    print(time()-btime)

    btime = time()
    for i in range(int(1e4)):
        for alpha in range(nel):
            for mode in range(nmodes):
                spfout = compute_meanfield_uncorr(alpha,mode,wf,ham.huterms,opspfs,opips)
    print(time()-btime)
