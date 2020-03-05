import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel,prange

class COOmat(object):
    def __init__(self, data, IA, JA):
        self.data = data
        self.IA = IA
        self.JA = JA
        self.nnz = self.IA[-1]
        self.nrows = len(self.IA)-1

class CSRmat(object):
    def __init__(self, data, IA, JA):
        self.data = data
        self.IA = IA
        self.JA = JA
        self.nnz = self.IA[-1]
        self.nrows = len(self.IA)-1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _matvec_par(long nrows,long[::1] IA,long[::1] JA,double[::1] data,complex[::1] vec,complex[::1] outvec) nogil:
    """
    """
    cdef int i,j,ncol,col_ind,d_ind
    d_ind = 0
    for i in range(nrows):
        ncol = IA[i+1]-IA[i]
        for j in range(ncol):
            col_ind = JA[d_ind]
            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
            d_ind += 1
    # TODO parallelize this 
    #with nogil, parallel():
    #    for i in prange(nrows, nogil=True):
    #        ncol = IA[i+1]-IA[i]
    #        for j in range(ncol):
    #            col_ind = JA[d_ind]
    #            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
    #            d_ind += 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _matvec(long nrows,long[::1] IA,long[::1] JA,double[::1] data,complex[::1] vec,complex[::1] outvec) nogil:
    """
    """
    cdef int i,j,ncol,col_ind,d_ind
    d_ind = 0
    for i in range(nrows):
        ncol = IA[i+1]-IA[i]
        for j in range(ncol):
            col_ind = JA[d_ind]
            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
            d_ind += 1
    # TODO parallelize this 
    #with nogil, parallel():
    #    for i in prange(nrows, nogil=True):
    #        ncol = IA[i+1]-IA[i]
    #        for j in range(ncol):
    #            col_ind = JA[d_ind]
    #            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
    #            d_ind += 1
    return

def matvec(object mat, cnp.ndarray[complex, ndim=1, mode='c'] vec):
    """
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] outvec = np.zeros_like(vec, dtype=complex)
    _matvec(mat.nrows,mat.IA,mat.JA,mat.data,vec,outvec)
    return outvec

def make_coo(int n,cnp.ndarray[double, ndim=2] op):
    """Turns a nxn dense matrix to coo format.
    """
    cdef int i,j,nnz
    cdef list data = []
    cdef list IA = []
    cdef list JA = []
    cdef cnp.ndarray JA_tmp
    IA.append( 0 )
    for i in range(n):
        JA_tmp = np.nonzero(op[i,:])[0]
        nnz = len(JA_tmp)
        IA.append( nnz )
        JA.append( list(JA_tmp) )
        data.append( list(op[i,JA_tmp]) )
    cdef cnp.ndarray[double, ndim=1, mode='c'] _data = np.array(data)
    cdef cnp.ndarray[long, ndim=1, mode='c'] _IA = np.array(IA, dtype=int)
    cdef cnp.ndarray[list, ndim=1, mode='c'] _JA = np.array(JA, dtype=list)
    cdef object opout = COOmat(_data,_IA,_JA)
    return opout

def make_csr(int n,cnp.ndarray[double, ndim=2] op):
    """Turns a nxn dense matrix to csr format.
    """
    cdef int i,j,nnz,nnz_row
    cdef list data = []
    cdef list IA = []
    cdef list JA = []
    cdef cnp.ndarray JA_tmp
    IA.append( 0 )
    nnz = 0
    for i in range(n):
        JA_tmp = np.nonzero(op[i,:])[0]
        nnz += len(JA_tmp)
        IA.append( nnz )
        data += list(op[i,JA_tmp])
        JA += list(JA_tmp)
    cdef cnp.ndarray[double, ndim=1, mode='c'] _data = np.array(data)
    cdef cnp.ndarray[long, ndim=1, mode='c'] _IA = np.array(IA, dtype=int)
    cdef cnp.ndarray[long, ndim=1, mode='c'] _JA = np.array(JA, dtype=int)
    cdef object opout = CSRmat(_data,_IA,_JA)
    return opout
