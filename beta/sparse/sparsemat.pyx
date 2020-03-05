import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp


cdef extern from "spmv.h" nogil:
    void zspmv(int nrows,int * IA,int * JA,double * data,double complex * vec,
               double complex * out,int nthr)

class CSRmat(object):
    def __init__(self, data, IA, JA, nthreads=1):
        self.data     = data
        self.IA       = IA
        self.JA       = JA
        self.nnz      = self.IA[-1]
        self.nrows    = len(self.IA)-1
        self.nthreads = nthreads

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode='c'] matvec(object mat,complex[::1] vec):
    """
    """
    cdef int[::1] IA = mat.IA
    cdef int[::1] JA = mat.JA
    cdef double[::1] data = mat.data
    cdef cnp.ndarray[complex, ndim=1, mode='c'] outvec = np.zeros_like(vec, dtype=complex)
    zspmv(mat.nrows,&IA[0],&JA[0],&data[0],&vec[0],&outvec[0],mat.nthreads)
    return outvec

def dense_to_csr(int n,cnp.ndarray[double, ndim=2] op):
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
    cdef cnp.ndarray[int, ndim=1, mode='c'] _IA = np.array(IA, dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1, mode='c'] _JA = np.array(JA, dtype=np.intc)
    cdef object opout = CSRmat(_data,_IA,_JA,openmp.omp_get_num_threads())
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
    cdef cnp.ndarray[int, ndim=1, mode='c'] _IA = np.array(IA, dtype=np.intc)
    cdef cnp.ndarray[int, ndim=1, mode='c'] _JA = np.array(JA, dtype=np.intc)
    cdef object opout = CSRmat(_data,_IA,_JA,openmp.omp_get_num_threads())
    return opout
