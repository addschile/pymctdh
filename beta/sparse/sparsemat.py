import numpy as np
from numba import njit

class CSRmat(object):
    def __init__(self, data, IA, JA):
        self.data  = data
        self.IA    = IA
        self.JA    = JA
        self.nnz   = self.IA[-1]
        self.nrows = len(self.IA)-1

@njit(fastmath=True)
def matvec(nrows,IA,JA,data,vec,outvec):
    """
    """
    d_ind = 0
    for i in range(nrows):
        ncol = IA[i+1]-IA[i]
        for j in range(ncol):
            col_ind = JA[d_ind]
            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
            d_ind += 1
    return outvec

def matadd(nrows,op1,a,op2,b):
    """
    """
    if op1 is None:
        opout = deepcopy(op2)
        opout.data *= b
    else:
        data = []
        JA = []
        IA = [0]
        ind1 = 0
        ind2 = 0
        for i in range(nrows):
            op1_col = op1.JA[op1.IA[i]:op1.IA[i+1]]
            op2_col = op2.JA[op2.IA[i]:op2.IA[i+1]]
            inds = np.union1d(op1_col,op2_col)
            IA.append( IA[i]+len(inds) )
            for ind in inds:
                JA.append( ind )
                dat = 0.0
                if ind in op1_col:
                    dat += a*op1.data[ind1]
                    ind1 +=1
                if ind in op2_col:
                    dat += b*op2.data[ind2]
                    ind2 +=1
                data.append( dat )
        data  = np.array(data)
        IA    = np.array(IA, dtype=np.intc)
        JA    = np.array(JA, dtype=np.intc)
        opout = CSRmat(data, IA, JA)
    return opout

def kron(nrows1,IA1,JA1,data1,nrows2,IA2,JA2,data2):
    """
    """
    # output data
    data = []
    JA = []
    IA = [0]

    for i in range(nrows1):
        col10 = IA1[i]
        col1f = IA1[i+1]
        ncol1 = col1f-col10
        d1 = data1[col10:col1f]
        j1 = JA1[col10:col1f]
        for j in range(nrows2):
            col20 = IA2[j]
            col2f = IA2[j+1]
            ncol2 = col2f-col20
            d2 = data2[col20:col2f]
            j2 = JA2[col20:col2f]
            IA.append( IA[-1] + ncol1*ncol2 )
            for k in range(ncol1):
                for l in range(ncol2):
                    data.append( d1[k]*d2[l] )
                    JA.append( j1[k]*nrows2 + j2[l] )
    return CSRmat(np.array(data), np.array(IA, dtype=int), np.array(JA, dtype=int))

def dense_to_csr(op):
    """Turns a nxn dense matrix to csr format.
    """
    n = op.shape[0]
    data = []
    IA = []
    JA = []
    IA.append( 0 )
    nnz = 0
    for i in range(n):
        JA_tmp = np.nonzero(op[i,:])[0]
        nnz += len(JA_tmp)
        IA.append( nnz )
        data.extend( list(op[i,JA_tmp]) )
        JA.extend( list(JA_tmp) )
    return CSRmat(np.array(data),np.array(IA,dtype=int),np.array(JA,dtype=int))

def csr_to_dense(nrows,IA,JA,data):
    """Turns a nxn csr matrix to dense format.
    """
    opout = np.zeros((nrows,)*2,dtype=data.dtype)
    d_ind = 0
    for i in range(nrows):
        ncol = IA[i+1]-IA[i]
        for j in range(ncol):
            col_ind = JA[d_ind]
            opout[i,col_ind] = data[d_ind]
            d_ind += 1
    return opout
