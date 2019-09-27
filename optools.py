import numpy as np
from cy.wftools import spf_innerprod,overlap_matrices2,compute_projector
from functools import lru_cache
from tensorutils import matelcontract,atensorcontract

def isdiag(op):
    if op == '1':
        return True
    elif op == 'sz':
        return True
    elif op == 'sx':
        return False 
    elif op == 'sy':
        return False 
    elif ',' in op:
        op_ = op.split(',')
        if op_[0]==op_[1]:
            return True
        else:
            return False
    else:
        return ValueError('Invalid electronic operator type.')

@lru_cache(maxsize=None,typed=False)
def compute_el_mes(alpha,beta,op):
    """Function that computes the matrix element between electronic states
    for nonadiabatic mctdh dynamics.
    """
    if op == '1':
        if alpha == beta: return 1.0
        else: return 0.0
    elif op == 'sz':
        if alpha != beta: return 0.0
        elif alpha == beta:
            if alpha == 0: return 1.0
            elif alpha == 1: return -1.0
    elif op == 'sx':
        if alpha == beta: return 0.0
        else: return 1.0
    elif op == 'sy':
        if alpha == beta: return 0.0
        else:
            if alpha == 0: return -1.0j
            else: return 1.0j
    elif ',' in op:
        op_ = op.split(',')
        if int(op_[0])==alpha and int(op_[1])==beta: return 1.0
        else: return 0.0
    else:
        return ValueError('Invalid electronic operator type.')

def precompute_ops(nel,nmodes,nspfs,npbfs,spfstart,spfend,ops,pbfs,*spfs):
    """Precomputes actions of hamiltonian operators on the spfs and the inner
    products.
    """
    # compute actions of the operators on spfs
    opspfs = compute_opspfs(nel,nmodes,nspfs,npbfs,spfstart,
                            spfend,ops,pbfs,spfs[0])
    # now compute the matrices of their inner products
    opips = compute_opips(nel,nmodes,nspfs,npbfs,spfstart,
                          spfend,ops,opspfs,spfs[-1])
    return opspfs , opips

def compute_opspfs(nel,nmodes,nspfs,npbfs,spfstart,spfend,ops,pbfs,spfs):
    """Computes all the individual operators that act on the spfs
    """
    opspfs = []
    # loop over each electronic state
    for i in range(nel):
        opspfsel = []
        # loop over each mode
        for j in range(nmodes):
            nspf = nspfs[i,j]
            npbf = npbfs[j]
            opspfmode = {}
            ind0 = spfstart[i,j]
            indf = spfend[i,j]
            # loop over unique operators for the mode
            for k in range(len(ops[j])):
                # get spf for this mode/electronic state
                spf = spfs[i][ind0:indf]
                spf_tmp = np.zeros_like(spf, dtype=complex)
                ind = 0 
                for n in range(nspf):
                    spf_tmp[ind:ind+npbf] = pbfs[j].operate(spf[ind:ind+npbf],
                                                            ops[j][k])
                    ind += npbf
                opspfmode[ops[j][k]] = spf_tmp
            opspfsel.append( opspfmode )
        opspfs.append( opspfsel )
    return opspfs

def compute_opips(nel,nmodes,nspfs,npbfs,spfstart,spfend,ops,opspfs,spfs):
    """Computes all the individual operators innerproducts
    """
    opips = []
    for i in range(nel):
        opipsel_i = []
        for j in range(i,nel):
            opipsel_j = []
            # loop over each mode
            for k in range(nmodes):
                nspf_i = nspfs[i,k]
                nspf_j = nspfs[j,k]
                npbf = npbfs[k]
                ind0 = spfstart[i,k]
                indf = spfend[i,k]
                opipsmode = {}
                # loop over unique operators for the mode
                for l in range(len(ops[k])):
                    opipsmode[ops[k][l]] = spf_innerprod(nspf_i,nspf_j,npbf,
                                                         spfs[i][ind0:indf],
                                                         opspfs[j][k][ops[k][l]])
                opipsel_j.append( opipsmode )
            opipsel_i.append( opipsel_j )
        opips.append( opipsel_i )
    return opips

def newmatelterm(nmodes,alpha,beta,A,A_,opterm,opips,spfovs):
    # TODO make this a more descriptive comment
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products.
    """
    mel = compute_el_mes(alpha,beta,opterm['elop'])
    if mel != 0.0:
        mel *= opterm['coeff']
        if 'modes' in opterm:
            if alpha <= beta:
                A_ += mel*matelcontract(nmodes,
                    opterm['modes'],opterm['ops'],opips[alpha][beta-alpha],
                    A, spfovs=spfovs[alpha][beta-alpha])
            else:
                A_ += mel*matelcontract(nmodes,
                    opterm['modes'],opterm['ops'],opips[beta][alpha-beta],
                    A, spfovs=spfovs[beta][alpha-beta],
                    conj=True)
        else: # purely electronic operator
            if alpha <= beta:
                A_ += mel*matelcontract(nmodes,None,None,
                    opips[alpha][beta-alpha],A,
                    spfovs=spfovs[alpha][beta-alpha])
            else:
                A_ += mel*matelcontract(nmodes,None,None,
                    opips[beta][alpha-beta],A,
                    spfovs=spfovs[beta][alpha-beta],
                    conj=True)

def matelterm(nmodes,alpha,beta,A,A_,opterm,opips,spfovs):
    # TODO make this a more descriptive comment
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products.
    """
    mel = compute_el_mes(alpha,beta,opterm['elop'])
    if mel != 0.0:
        mel *= opterm['coeff']
        if 'modes' in opterm:
            if alpha == beta:
                A_ += mel*matelcontract(nmodes,
                    opterm['modes'],opterm['ops'],opips[alpha][beta-alpha],
                    A)
            elif alpha < beta:
                A_ += mel*matelcontract(nmodes,
                    opterm['modes'],opterm['ops'],opips[alpha][beta-alpha],
                    A, spfovs=spfovs[alpha][beta-alpha-1])
            elif alpha > beta:
                A_ += mel*matelcontract(nmodes,
                    opterm['modes'],opterm['ops'],opips[beta][alpha-beta],
                    A, spfovs=spfovs[beta][alpha-beta-1],
                    conj=True)
        else: # purely electronic operator
            if alpha == beta:
                A_ += mel*matelcontract(nmodes,None,None,
                    opips[alpha][beta-alpha],A)
            elif alpha < beta:
                A_ += mel*matelcontract(nmodes,None,None,
                    opips[alpha][beta-alpha],A,
                    spfovs=spfovs[alpha][beta-alpha-1])
            elif alpha > beta:
                A_ += mel*matelcontract(nmodes,None,None,
                    opips[beta][alpha-beta],A,
                    spfovs=spfovs[beta][alpha-beta-1],
                    conj=True)

def matel(nel,nmodes,nspfs,npbfs,opterms,opips,spfovs,A):
    # TODO make this a more descriptive comment
    """Compute matrix elements acting on the A tensor from a precomputed set of
    spf inner products.
    """
    Aout = np.zeros(nel, dtype=np.ndarray)
    # loop over the electronic states
    for alpha in range(nel):
        A_ = np.zeros_like(A[alpha], dtype=complex)
        for beta in range(nel):
            # loop over the terms in the Hamiltonian
            for opterm in opterms:
                matelterm(nmodes,alpha,beta,A[beta],A_,opterm,opips,spfovs)
        Aout[alpha] = A_
    return Aout

def act_operator(wf,op,tol=1.e-8,maxiter=100):
    """
    """
    from wavefunction import Wavefunction
    from meanfield import compute_meanfield_uncorr
    # get wf info
    nel = wf.nel
    nmodes = wf.nmodes
    nspf = wf.nspf
    npbf = wf.npbf
    spfstart = wf.spfstart
    spfend = wf.spfend

    # create wavefunction data for iterations
    A_k      = wf.copy('A')
    spfs_k   = wf.copy('spfs')
    A_kp1    = wf.copy('A')
    spfs_kp1 = wf.copy('spfs')

    # compute action of operator on wf spfs
    # TODO arguments
    opspfs = compute_opspfs(op.ops,wf)
    opips = compute_opips(op.ops,wf,opspfs)
    #return opspfs,opips

    # compute overlap matrices
    # TODO arguments
    spfovs = overlap_matrices2(wf,wf)
    #return spfovs

    # update initial A tensor
    for alpha in range(nel):
        A_k[alpha] *= 0.0
        for beta in range(nel):
            # TODO arguments
            newmatelterm(nmodes,alpha,beta,wf.A[beta],A_k[alpha],op.term,opips,spfovs)
    #return wf_k.A

    # perform iterations until convergence is reached
    flag = 1
    for i in range(maxiter):
        # update spfs
        if 'modes' in op.term:
            for alpha in range(nel):
                spfs_kp1[alpha] *= 0.0
                for mode in range(nmodes):
                    ind0 = spfstart[alpha,mode]
                    indf = spfend[alpha,mode]
                    # TODO arguments
                    spfs_kp1[alpha][ind0:indf] += compute_meanfield_uncorr(alpha,mode,wf,[op.term],opspfs,gen=True)
            # gram-schmidt orthogonalize the spfs
            # TODO arguments
            wf_kp1.orthonormalize_spfs()
        #return wf_kp1
        # update the A tensor coefficients
        # TODO arguments
        opips = compute_opips(op.ops,wf_kp1,opspfs)
        spfovs = overlap_matrices2(wf_kp1,wf)
        for alpha in range(nel):
            A_kp1[alpha] *= 0.0
            for beta in range(nel):
                # TODO arguments
                newmatelterm(nmodes,alpha,beta,wf.A[beta],wf_kp1.A[alpha],op.term,opips,spfovs)
        # check convergence
        delta = 0.0
        for alpha in range(nel):
            for mode in range(nmodes):
                ind0 = spfstart[alpha,mode]
                indf = spfend[alpha,mode]
                # compute criteria based on projectors
                p_k = compute_projector(nspf[alpha,mode],npbf[mode],spfs_k.spfs[alpha][ind0:indf])
                p_kp1 = compute_projector(nspf[alpha,mode],npbf[mode],spfs_kp1.spfs[alpha][ind0:indf])
                delta += np.linalg.norm(p_kp1-p_k)
        if delta < tol:
            flag = 0
            break
        else:
            A_k    = deepcopy(A_kp1)
            spfs_k = deepcopy(spfs_kp1)

    if flag:
        raise ValueError("Maximum iterations reached. Could not converge.")

    # copy data and normalize wavefunction
    wf.A = deepcopy(A_kp1)
    wf.spfs = deepcopy(spfs_kp1)
    wf.normalize()

def compute_expect(op,wf,pbfs):
    """Computes the expectation value of a generic operator.
    """
    # get wf info
    nmodes = wf.nmodes
    nel = wf.nel
    nspfs = wf.nspfs
    npbfs = wf.npbfs

    modes = op.term['modes']
    mode = op.term['modes'][0]
    _op = op.term['ops'][0]
    # make matrix of spf inner products
    opspfs,opips = precompute_ops(nmodes,nel,nspfs,npbfs,spfstart,spfend,op.ops,pbfs,wf.spfs)
    # contract (A*)xA with opips
    expect = 0.0
    for alpha in range(wf.nel):
        Asum = atensorcontract(modes, wf.A[alpha])
        expect += np.einsum('ij,ij',Asum,opips[alpha][0][mode][_op]).real
    return expect

def jump_probs(LdLs,wf,pbfs):
    """Computes the jump probabilities of all the quantum jump operators.
    """
    p_n = np.zeros(len(LdLs))
    for i in range(len(LdLs)):
        p_n[i] = compute_expect(LdLs[i],wf,pbfs)
    p = np.sum(p_n)
    return p_n , p

def jump(rand,Ls,LdLs,wf,pbfs):
    """
    """
    # compute jump probabilities
    p_n , p = jump_probs(LdLs,wf,pbfs)
    # see which one it jumped along
    p *= rand
    for count in range(len(Ls)):
        if p <= np.sum(p_n[:count+1]):
            act_operator(wf,Ls[count]) , count
            return

if __name__ == "__main__":

    from wavefunction import Wavefunction
    from hamiltonian import Hamiltonian
    from qoperator import QOperator
    from eom import eom_coeffs
    from meanfield import compute_meanfield_uncorr

    nel    = 2
    nmodes = 4
    nspf = np.array([[8, 13, 7, 6],
                     [7, 12, 6, 5]], dtype=int)
    npbf = np.array([22, 32, 21, 12], dtype=int)

    pbfs = list()
    pbfs.append( PBasis(['ho', 22, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 32, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 21, 1.0, 1.0]) )
    pbfs.append( PBasis(['ho', 12, 1.0, 1.0]) )

    wf = Wavefunction(nel, nmodes, nspf, npbf)
    wf.generate_ic(1)
    wf.overlap_matrices()

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

    ham = Hamiltonian(nmodes, hterms, wf=wf)
    #print(ham.ops)
    #for i in range(nmodes):
    #    print(wf.pbfs[i].ops.keys())

    # precompute action of operators and inner products
    print('precompute_ops')
    opspfs,opips = precompute_ops(ham.ops, wf)
    #print(opips)

    term = {'coeff': 0.01, 'units': 'ev', 'modes': 0, 'ops': 'q'}
    #print('with hamiltonian first')
    #ham = Hamiltonian(nmodes, [term.copy()], wf=wf)
    #opspfs,opips = precompute_ops(ham.ops, wf)
    ##Aout2 = matel(wf,ham.hterms,opips)
    #Aout = []
    #for alpha in range(nel):
    #    A_ = np.zeros_like(wf.A[alpha], dtype=complex)
    #    for beta in range(nel):
    #        matelterm(nmodes,alpha,beta,wf.A[beta],A_,ham.hterms[0],opips,wf.spfovs)
    #    Aout.append(A_)
    #for i in range(nel):
    #    print(np.allclose(Aout[i],Aout2[i]))
    #wf_kp1 = wf.copy()
    #for alpha in range(nel):
    #    wf_kp1.spfs[alpha] *= 0.0
    #    for mode in range(nmodes):
    #        ind0 = wf.spfstart[alpha,mode]
    #        indf = wf.spfend[alpha,mode]
    #        wf_kp1.spfs[alpha][ind0:indf] += compute_meanfield_uncorr(alpha,mode,wf,ham.hterms,opspfs,gen=True)

    op = QOperator(nmodes, term, wf=wf)
    #for i in range(nmodes):
    #    print(wf.pbfs[i].ops.keys())

    ## check to make sure opspfs and opips are the same
    #opspfs2,opips2 = act_operator(wf,op)
    #print('opspfs')
    #for i in range(nel):
    #    print(i)
    #    print(np.allclose(opspfs[i][0]['q'],opspfs2[i][0]['q']))
    #for i in range(nel):
    #    print(i)
    #    for j in range(1,nmodes):
    #        print(opspfs[i][j],opspfs2[i][j])
    #print('')
    #print('opips')
    #for i in range(nel):
    #    for j in range(nel-i):
    #        print(i,j)
    #        print(np.allclose(opips[i][j][0]['q'],opips2[i][j][0]['q']))
    #for i in range(nel):
    #    for j in range(nel-i):
    #        print(i,j)
    #        for k in range(1,nmodes):
    #            print(opips[i][j][k],opips2[i][j][k])

    ## check to make sure ovs are the same
    #spfovs = act_operator(wf,op)
    #for i in range(nel):
    #    for j in range(i,nel):
    #        print(i,j)
    #        for k in range(nmodes):
    #            ov = spfovs[i][j-i][k]
    #            if i==j:
    #                print(np.allclose(np.eye(len(ov)),ov))
    #            else:
    #                print(np.allclose(wf.spfovs[i][j-i-1][k],ov))

    ## check to make sure new A tensor is correct
    #new_A = act_operator(wf,op)
    #for i in range(nel):
    #    print(np.sum(Aout[i]),np.sum(new_A[i]))
    #    print(np.allclose(Aout[i],new_A[i]))

    ## check to make sure mean field operation is correct
    #new_wf = act_operator(wf,op)
    #for alpha in range(nel):
    #    print(np.allclose(wf_kp1.spfs[alpha],new_wf.spfs[alpha]))

    # act this new operator on the wavefunction
    print('act op')
    new_wf = act_operator(wf,op)
    #for i in range(nel):
    #    print(new_wf.spfs[i])
    # compute the energy of this new wavefunction
    wf.overlap_matrices()
    print('precompute_ops')
    opspfs,opips = precompute_ops(ham.ops, wf)
    A = eom_coeffs(wf, ham, opips)
    energy = 0.0
    for i in range(nel):
        energy += (1.j*np.sum(wf.A[i].conj()*A[i])).real
    print(energy/0.0367493)
    new_wf.overlap_matrices()
    print('precompute_ops')
    opspfs,opips = precompute_ops(ham.ops, new_wf)
    A = eom_coeffs(new_wf, ham, opips)
    energy = 0.0
    for i in range(nel):
        energy += (1.j*np.sum(new_wf.A[i].conj()*A[i])).real
    print(energy/0.0367493)
