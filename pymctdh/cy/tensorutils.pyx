import numpy as np
cimport numpy as cnp

def matelcontract(int nmodes,list modes,opips,cnp.ndarray A,spfovs=None,conj=False):
    cdef int i
    cdef cnp.ndarray output
    cdef int count
    cdef int opcount
    cdef list skipped
    cdef list unskipped
    cdef list outorder
    cdef list outaxes
    if modes == None: # purely electronic operator
        if spfovs == None:
            # no contraction should occur
            return A
        else:
            output = A
            for i in range(nmodes):
                if conj:
                    output = np.tensordot(output,spfovs[i].conj(),axes=[[0],[0]])
                else:
                    output = np.tensordot(output,spfovs[i],axes=[[0],[1]])
            return output
    elif len(modes) == 1 and spfovs is None:
        # for umatelterms
        output = np.tensordot(A,opips[0],axes=[[modes[0]],[1]])
        outaxes = [i for i in range(nmodes)]
        outaxes = outaxes[:modes[0]] + [outaxes[-1]] + outaxes[modes[0]:-1]
        return np.transpose(output, axes=outaxes)
    else:
        # for umatelterms
        output = A
        count = 0
        opcount = 0
        skipped = []
        unskipped = []
        for i in range(nmodes):
            if i in modes:
                unskipped.append( i )
                if conj:
                    output = np.tensordot(output,opips[opcount].conj(),axes=[[count],[0]])
                else:
                    output = np.tensordot(output,opips[opcount],axes=[[count],[1]])
                opcount += 1
            else:
                if spfovs != None:
                    unskipped.append( i )
                    if conj:
                        output = np.tensordot(output,spfovs[i].conj(),axes=[[count],[0]])
                    else:
                        output = np.tensordot(output,spfovs[i],axes=[[count],[1]])
                else:
                    skipped.append(i)
                    count += 1
        outorder = [skipped[i] for i in range(len(skipped))]
        outorder += [unskipped[i] for i in range(len(unskipped))]
        outaxes = [outorder.index(i) for i in range(len(outorder))]
        return np.transpose(output, axes=outaxes)

def atensorcontract(int nmodes, list modes, *As, spfovs=None, spfovsconj=False):
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
    cdef int i
    cdef list path
    cdef cnp.ndarray output
    cdef int count
    cdef list path1
    cdef list path2
    # generate the initial list of indices, which will all be independent
    if spfovs is None:
        path = [i for i in range(nmodes)]
        for i in range(len(modes)):
            path.remove(modes[i])
        if len(As) == 1:
            return np.tensordot(As[0].conj(),As[0],axes=[path,path])
        elif len(As) == 2:
            return np.tensordot(As[0].conj(),As[1],axes=[path,path])
    elif isinstance(spfovs,list):
        output = As[0].conj()
        count = 0
        path1 = []
        path2 = []
        for i in range(nmodes):
            if not i in modes:
                path1.append(i+len(modes)-count)
                path2.append(i)
                if spfovsconj:
                    output = np.tensordot(output,spfovs[i].conj(),axes=[[count],[1]])
                else:
                    output = np.tensordot(output,spfovs[i],axes=[[count],[0]])
            else:
                count += 1
        return np.tensordot(output,As[1],axes=[path1,path2])
    else:
        output = As[0].conj()
        count = 0
        path1 = []
        path2 = []
        for i in range(nmodes):
            if not i in modes:
                path1.append(i+len(modes)-count)
                path2.append(i)
                if spfovsconj:
                    output = np.tensordot(output,spfovs.conj(),axes=[[count],[1]])
                else:
                    output = np.tensordot(output,spfovs,axes=[[count],[0]])
            else:
                count += 1
        return np.tensordot(output,As[1],axes=[path1,path2])

def ahtensorcontract(int nmodes,cnp.ndarray A,cnp.ndarray[complex, ndim=2, mode='c'] h,int order,conj=False):
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
    cdef int nmodes_ = int(nmodes/2)
    cdef list path
    if order == 0:
        if conj:
            path = [nmodes_,0]
        else:
            path = [0,nmodes_]
    else:
        if conj:
            path = [nmodes_+1,1]
        else:
            path = [1,nmodes_+1]
    return np.tensordot(A,h,axes=[path,[0,1]])
