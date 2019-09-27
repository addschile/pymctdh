import numpy as np

def matelcontract(nmodes,modes,ops,opips,A, spfovs=None, conj=False):
    """Computes the contractions required for mctdh coefficient eom.

    Input
    -----
    nmodes - int, number of modes in the system
    modes - list, the modes that are acted on by the hamiltonian term
    ops - list, the operators that act on the modes in the hamiltonian term
    opips - list, the list/dictionary structure that stores the inner products
            of the hamiltonian terms for each spf
    A - np.ndarray, nmode-dimensional the mctdh A tensor for an electronic state
    spfovs - the spf overlap matrices between different electronic states

    Output
    ------
    np.ndarray - nmode-dimensional tensor, this should have the same shape as A

    Notes
    -----
    """
    # generate the initial list of indices, which will all be independent
    path1 = [i for i in range(nmodes)]
    # arrays and indices to contract over as an ordered tuple
    arglist = [A,path1]
    opcount = 0
    indcount = nmodes
    outorder = []
    if modes == None: # purely electronic operator
        if spfovs == None:
            # no contraction should occur
            return A
        else:
            for i in range(nmodes):
                arglist.append( spfovs[i] )
                arglist.append( [indcount,i] )
                outorder.append( indcount )
                indcount += 1
    else:
        for i in range(nmodes):
            if i in modes:
                if conj:
                    arglist.append( opips[i][ops[opcount]].conj() )
                    arglist.append( [i,indcount] )
                else:
                    arglist.append( opips[i][ops[opcount]] )
                    arglist.append( [indcount,i] )
                outorder.append( indcount )
                opcount += 1
                indcount += 1
            else:
                if spfovs != None:
                    if conj:
                        arglist.append( spfovs[i].conj() )
                        arglist.append( [i,indcount] )
                    else:
                        arglist.append( spfovs[i] )
                        arglist.append( [indcount,i] )
                    outorder.append( indcount )
                    indcount += 1
                else:
                    outorder.append( i )
    # perform the contraction and return
    return np.einsum(*arglist, outorder, order='C', optimize='greedy')

def atensorcontract(modes, *As, spfovs=None, spfovsconj=False):
    """Function that does the contraction over MCTDH A tensor when computing 
    mean field operators.

    Indices for the electronic variables are taken care of outside of this 
    function.

    Input
    -----
    modes - list, of modes that I do not contract over
    As - np.ndarray(s), the mctdh A tensor(s for different electronic states)
    spfovs - the spf overlap matrices between different electronic states

    Output
    ------
    np.ndarray - the tensor of coefficients that were not contracted over

    Notes
    -----
    For a one-body operator that acts on mode k, this function will return a
    matrix of coefficients that are then multiplied by each matrix element for
    the one-body operator. If we call this output array O, then this will be 
    used to compute mean field operators like

    <h(t)>^k_{nm} = O_{nm} <\phi^k_n (t)| h^1_k |\phi^k_m (t)>

    If we have a two-body operator that acts on mode k and then on mode l, then
    this function will return a 4-dimensional tensor to compute mean field 
    operators like

    <h(t)>^k_{nm} = <\phi^k_n (t)| h^1_k |\phi^k_m (t)> x
                      \sum_{pq} O_{nmpq} <\phi^l_p (t)| h^1_l |\phi^k_q (t)>
    """
    A = As[0]
    nmodes = len(A.shape)
    indcount = nmodes
    # generate the initial list of indices, which will all be independent
    path1 = [i for i in range(nmodes)]
    if spfovs == None:
        # this is when alpha == beta
        if len(modes) == 0:
            path2 = path1
        else:
            count = 0
            path2 = []
            for i in range(nmodes):
                if i in modes:
                    path2.append( indcount )
                    indcount += 1
                else:
                    path2.append( path1[i] )
        if len(As) == 1:
            return np.einsum(A.conj(),path1,A,path2, order='C', optimize='greedy')
        elif len(As) == 2:
            return np.einsum(A.conj(),path1,As[1],path2, order='C', optimize='greedy')
    else:
        path2 = [i+indcount for i in range(nmodes)]
        arglist = [A.conj(), path1, As[1], path2]
        for i in range(nmodes):
            if not i in modes:
                if spfovsconj:
                    arglist.append( spfovs[i].conj() )
                    arglist.append( [path2[i],path1[i]] )
                else:
                    arglist.append( spfovs[i] )
                    arglist.append( [path1[i],path2[i]] )
        return np.einsum(*arglist, order='C', optimize='greedy')

def ahtensorcontract(A,h,order,conj=False):
    """Function that does the contraction over remaining MCTDH A tensor with 
    hamiltonian term matrix elements mwhen computing mean field operators.

    Input
    -----
    A - np.ndarray, the mctdh A tensor that has been contracted over all modes
    that aren't a part of this meanfield operator
    h - np.ndarray, the matrix elements of the hamiltonian term to be 
    contracted over

    Output
    ------
    np.ndarray - the tensor of coefficients that were not contracted over,
    should be a tensor with 2 fewer dimensions than what came in
    """
    nmodes = len(A.shape)
    path1  = [i for i in range(nmodes)]
    nmodes = int(nmodes/2)
    if order == 0:
        if conj:
            path2 = [path1[nmodes],path1[0]]
        else:
            path2 = [path1[0],path1[nmodes]]
    else:
        if conj:
            path2 = [path1[nmodes+1],path1[1]]
        else:
            path2 = [path1[1],path1[nmodes+1]]
    return np.einsum(A,path1,h,path2, order='C', optimize='greedy')
