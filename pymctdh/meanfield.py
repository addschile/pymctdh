import numpy as np
from pymctdh.optools import compute_el_mes
from pymctdh.cy.wftools import spf_innerprod,multAtens
from pymctdh.cy.tensorutils import atensorcontract,ahtensorcontract

# TODO this needs to be generalized
def compute_meanfield_uncorr_op(nel, nmodes, nspf, npbf, spfstart, spfend, 
                                opterm, uopspfs, spfs, spfsout):
    """
    """
    for alpha in range(nel):
        spf    = spfs[alpha]
        spfout = spfsout[alpha]
        opspfs = uopspfs[alpha]
        if 'modes' in opterm:
            for mode in range(nmodes):
                ind0 = spfstart[alpha,mode]
                indf = spfend[alpha,mode]
                if mode == opterm['modes'][0]:
                    spfout[ind0:indf] += opspfs[ind0:indf]
                else:
                    spfout[ind0:indf] += spf[ind0:indf]
        else:
            spfout += opterm['coeff']*spfs

def compute_meanfield_uncorr(nel, nmodes, spfstart, spfend, uopspfs, spfs, spfsout):
    """Computes the meanfield operators for uncorrelated hamiltonian terms.
    Input
    -----
    nel -
    nmodes -
    uopspfs -
    spfs -
    spfsout -
    """
    for alpha in range(nel):
        spfsout[alpha] += uopspfs[alpha]

# TODO incorporate this into _uncorr function
def compute_meanfield_eluncorr(nel, nmodes, spfstart, spfend, huelterms, spfs, spfsout):
    """Computes the meanfield operators for uncorrelated hamiltonian terms.
    Input
    -----
    nel -
    nmodes -
    uopspfs -
    spfs -
    spfsout -
    """
    for hterm in huelterms:
        elop  = hterm['elop']
        coeff = hterm['coeff']
        for alpha in range(nel):
            spfout = spfsout[alpha]
            spf = spfs[alpha]
            mel = compute_el_mes(alpha,alpha,elop)
            if mel != 0.0:
                spfout += mel*coeff*spf

def compute_meanfield_corr(nel, nmodes, nspfs, npbfs, spfstart, spfend, copspfs,
                           copips, spfovs, A, spfs, spfsout):
    """Computes the meanfield operators and their action on the spfs.
    Meanfield operators are stored if using the CMF integrator.

    Input
    -----
    nel -
    nmodes -
    nspfs
    npbfs
    spfstart
    spfend
    copspfs
    copips
    spfovs
    A
    spfs
    spfout - np.ndarray, output of meanfield operators acting on input spf

    Notes
    -----
    For each term in the Hamiltonian the order of operators for how the
    mean field operator acts on the spfs is:
    1) contract the A tensor over all modes that aren't acted on by the term
    2) compute the inner products of all the spfs that are acted on by the term
    3) contract over the contracted A tensor and the matrix elements of the modes
    4) act the meanfield operator on the vector of spfs
    """
    # loop over the correlated terms
    for i,cip in enumerate(copips):
        modes = cip['modes']
        elop  = cip['elop']
        coeff = cip['coeff']
        opips = cip['opips']
        for alpha in range(nel):
            for beta in range(nel):
                mel = compute_el_mes(alpha,beta,elop)
                if mel != 0.0:
                    mel *= coeff
                    for mode in range(nmodes):
                        nspf_l  = nspfs[alpha,mode]
                        nspf_r  = nspfs[beta,mode]
                        npbf    = npbfs[mode]
                        ind0_l  = spfstart[alpha,mode]
                        indf_l  = spfend[alpha,mode]
                        ind0_r  = spfstart[beta,mode]
                        indf_r  = spfend[beta,mode]
                        spfout_ = spfsout[alpha][ind0_l:indf_l]
                        if mode not in modes:
                            hmodes = modes + [mode]
                            hmodes.sort()
                        else:
                            hmodes = modes.copy()
                            hmodes.sort()
                        if len(hmodes) != nmodes:
                            if alpha == beta:
                                Asum = atensorcontract(nmodes, hmodes, A[beta])
                            elif alpha < beta:
                                Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                       spfovs=spfovs[alpha][beta-alpha-1])
                            else:
                                Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                       spfovs=spfovs[beta][alpha-beta-1],
                                                       spfovsconj=True)
                            # contract over inner products of ham terms of other modes
                            for num,hmode in enumerate(modes):
                                if hmode != mode:
                                    if alpha <= beta:
                                        opip = opips[alpha][beta][num]
                                        conj = False
                                    else:
                                        opip = opips[beta][alpha][num].conj()
                                        conj = True
                                    if hmode < mode:
                                        order = 0
                                    else:
                                        order = 1
                                    Asum = ahtensorcontract(nmodes,Asum,opip,order,conj=conj)
                            if mode in modes:
                                spfout_ += mel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                     copspfs[i][beta][ind0_r:indf_r])
                            else:
                                spfout_ += mel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                     spfs[beta][ind0_r:indf_r])
                        else:
                            for num,hmode in enumerate(modes):
                                if hmode != mode:
                                    hmodes.remove( hmode )
                                    if alpha <= beta:
                                        opip = opips[alpha][beta][num]
                                        conj = False
                                    else:
                                        opip = opips[beta][alpha][num]
                                        conj = True
                                    if hmode < mode:
                                        order = 0
                                    else:
                                        order = 1
                                    Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                           spfovs=opip, spfovsconj=conj)
                            if mode in modes:
                                spfout_ += mel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                     copspfs[i][beta][ind0_r:indf_r])
                            else:
                                spfout_ += mel*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                     spfs[beta][ind0_r:indf_r])
    return spfsout

def compute_meanfield_elcorr(nel, nmodes, nspfs, npbfs, spfstart, spfend, hcelterms,
                           spfovs, A, spfs, spfsout):
    """Computes the meanfield operators and their action on the spfs.
    Meanfield operators are stored if using the CMF integrator.

    Input
    -----
    nel -
    nmodes -
    nspfs
    npbfs
    spfstart
    spfend
    copspfs
    copips
    spfovs
    A
    spfs
    spfout - np.ndarray, output of meanfield operators acting on input spf

    Notes
    -----
    For each term in the Hamiltonian the order of operators for how the
    mean field operator acts on the spfs is:
    1) contract the A tensor over all modes that aren't acted on by the term
    2) compute the inner products of all the spfs that are acted on by the term
    3) contract over the contracted A tensor and the matrix elements of the modes
    4) act the meanfield operator on the vector of spfs
    """
    for hterm in hcelterms:
        elop  = hterm['elop']
        coeff = hterm['coeff']
        for alpha in range(nel):
            spfout = spfsout[alpha]
            for beta in range(nel):
                spf = spfs[beta]
                mel = compute_el_mes(alpha,beta,elop)
                if mel != 0.0:
                    for mode in range(nmodes):
                        # contract over all other modes
                        if alpha == beta:
                            ovs  = None
                            conj = False
                        elif alpha < beta:
                            ovs  = spfovs[alpha][beta-alpha-1]
                            conj = False
                        else:
                            ovs  = spfovs[beta][alpha-beta-1]
                            conj = True 
                        Asum = atensorcontract(nmodes, [mode], A[alpha], A[beta],
                                               spfovs=ovs, spfovsconj=conj)
                        nspf_l = nspfs[alpha,mode]
                        nspf_r = nspfs[beta,mode]
                        npbf   = npbfs[mode]
                        ind0_l = spfstart[alpha,mode]
                        indf_l = spfend[alpha,mode]
                        ind0_r = spfstart[beta,mode]
                        indf_r = spfend[beta,mode]
                        spfout[ind0_l:indf_l] += mel*coeff*multAtens(nspf_l,nspf_r,npbf,Asum,
                                                 spf[ind0_r:indf_r])

def act_meanfield(nel, nmodes, nspfs, npbfs, spfstart, spfend, mfs, copspfs, 
                  copips, spfs, spfsout):
    """Computes the action of the meanfield operators on the spfs from stored
    meanfield operators. This is only used in the CMF integration scheme.
    """
    mfcount = 0
    for i,cip in enumerate(copips):
        modes = cip['modes']
        elop  = cip['elop']
        coeff = cip['coeff']
        opips = cip['opips']
        for alpha in range(nel):
            for beta in range(nel):
                mel = compute_el_mes(alpha,beta,elop)
                if mel != 0.0:
                    for mode in range(nmodes):
                        # get array sizes
                        nspf_l = nspfs[alpha,mode]
                        nspf_r = nspfs[beta,mode]
                        npbf   = npbfs[mode]
                        # get indices
                        ind0_l = spfstart[alpha,mode]
                        indf_l = spfend[alpha,mode]
                        ind0_r = spfstart[beta,mode]
                        indf_r = spfend[beta,mode]
                        # get mf operators
                        mf       = mfs[mfcount]
                        mfcount += 1
                        # get output
                        spfout = spfsout[alpha][ind0_l:indf_l]
                        if mode in modes:
                            # apply things that need underlying pbf operation
                            spfout += multAtens(nspf_l,nspf_r,npbf,mf,
                                                copspfs[i][beta][ind0_r:indf_r])
                        else:
                            # apply things that don't need underlying pbf 
                            # operation
                            spfout += multAtens(nspf_l,nspf_r,npbf,mf,
                                                spfs[beta][ind0_r:indf_r])
    return spfsout

def compute_meanfield_mats(nel, nmodes, nspfs, npbfs, spfstart, spfend, copips,
                           spfovs, A):
    """
    """
    mfs = []
    # loop over the correlated terms
    for i,cip in enumerate(copips):
        modes = cip['modes']
        elop  = cip['elop']
        coeff = cip['coeff']
        opips = cip['opips']
        for alpha in range(nel):
            for beta in range(nel):
                mel = compute_el_mes(alpha,beta,elop)
                if mel != 0.0:
                    mel *= coeff
                    for mode in range(nmodes):
                        nspf_l  = nspfs[alpha,mode]
                        nspf_r  = nspfs[beta,mode]
                        npbf    = npbfs[mode]
                        ind0_l  = spfstart[alpha,mode]
                        indf_l  = spfend[alpha,mode]
                        ind0_r  = spfstart[beta,mode]
                        indf_r  = spfend[beta,mode]
                        if mode not in modes:
                            hmodes = modes + [mode]
                            hmodes.sort()
                        else:
                            hmodes = modes.copy()
                            hmodes.sort()
                        if len(hmodes) != nmodes:
                            if alpha == beta:
                                Asum = atensorcontract(nmodes, hmodes, A[beta])
                            elif alpha < beta:
                                Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                       spfovs=spfovs[alpha][beta-alpha-1])
                            else:
                                Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                       spfovs=spfovs[beta][alpha-beta-1],
                                                       spfovsconj=True)
                            # contract over inner products of ham terms of other modes
                            for num,hmode in enumerate(modes):
                                if hmode != mode:
                                    if alpha <= beta:
                                        opip = opips[alpha][beta][num]
                                        conj = False
                                    else:
                                        opip = opips[beta][alpha][num].conj()
                                        conj = True
                                    if hmode < mode:
                                        order = 0
                                    else:
                                        order = 1
                                    Asum = ahtensorcontract(nmodes,Asum,opip,order,conj=conj)
                            mfs.append( mel*Asum )
                        else:
                            for num,hmode in enumerate(modes):
                                if hmode != mode:
                                    hmodes.remove( hmode )
                                    if alpha <= beta:
                                        opip = opips[alpha][beta][num]
                                        conj = False
                                    else:
                                        opip = opips[beta][alpha][num]
                                        conj = True
                                    if hmode < mode:
                                        order = 0
                                    else:
                                        order = 1
                                    Asum = atensorcontract(nmodes, hmodes, A[alpha], A[beta], 
                                                           spfovs=opip, spfovsconj=conj)
                            mfs.append( mel*Asum )
    return mfs 

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

#    # TODO update this
#    wf.overlap_matrices()
#    opspfs,opips = precompute_ops(ham.ops, wf)
#
#    btime = time()
#    for i in range(int(1e3)):
#        for alpha in range(nel):
#            for beta in range(nel):
#                spfout = compute_meanfield_corr(alpha,beta,wf,ham.hcterms,opspfs,opips)
#    print(time()-btime)
#
#    btime = time()
#    for i in range(int(1e4)):
#        for alpha in range(nel):
#            for mode in range(nmodes):
#                spfout = compute_meanfield_uncorr(alpha,mode,wf,ham.huterms,opspfs,opips)
#    print(time()-btime)
