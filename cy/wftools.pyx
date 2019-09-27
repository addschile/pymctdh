import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spf_innerprod(int nspf_l,int nspf_r,int npbf,complex[::1] spfs1,complex[::1] spfs2,complex[:,::1] spfsout) nogil:
    """
    """
    cdef int i,j
    cdef int ind_i,ind_j
    cdef int incx = 1
    cdef int incy = 1
    ind_i = 0
    for i in range(nspf_l):
        ind_j = 0
        for j in range(nspf_r):
            spfsout[i,j] = blas.zdotc(&npbf,&spfs1[ind_i:ind_i+npbf][0],&incx,&spfs2[ind_j:ind_j+npbf][0],&incy)
            ind_j += npbf
        ind_i += npbf
    return

def spf_innerprod(int nspf_l,int nspf_r,int npbf,cnp.ndarray[complex, ndim=1, mode='c'] spfs1,cnp.ndarray[complex, ndim=1, mode='c'] spfs2):
    """Computes the inner product of two single-particle functions.
    """
    if nspf_l==nspf_r:
        if np.linalg.norm((spfs1-spfs2))<1.e-8:
            return np.eye(nspf_l, dtype=complex)
    cdef cnp.ndarray[complex, ndim=2, mode='c'] spfsout = np.zeros((nspf_l,nspf_r),dtype=complex)
    _spf_innerprod(nspf_l,nspf_r,npbf,spfs1,spfs2,spfsout)
    return spfsout

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _multAtens(int nspf_l,int nspf_r,int npbf,complex[:,:] A,complex[::1] spfs,complex[::1] spfsout) nogil:
    """Computes the multiplication of the A tensor with the single-particle 
       functions.
    """
    cdef int i,j
    cdef int ind_i,ind_j
    cdef int incx = 1
    cdef int incy = 1
    ind_i = 0
    for i in range(nspf_l):
        ind_j = 0
        for j in range(nspf_r):
            blas.zaxpy(&npbf,&A[i,j],&spfs[ind_j:ind_j+npbf][0],&incx,&spfsout[ind_i:ind_i+npbf][0],&incy)
            ind_j += npbf
        ind_i += npbf
    return

def multAtens(int nspf_l,int nspf_r,int npbf,cnp.ndarray[complex, ndim=2] A,cnp.ndarray[complex, ndim=1, mode='c'] spfs):
    """Computes the multiplication of the A tensor with the single-particle 
       functions.
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] spfsout = np.zeros(nspf_l*npbf, dtype=complex)
    _multAtens(nspf_l,nspf_r,npbf,A,spfs,spfsout)
    return spfsout

def compute_density_matrix(int nspf,int alpha,int mode,cnp.ndarray A):
    """Computes density matrix for given mode and electronic states.
    """
    cdef int i,j
    cdef cnp.ndarray[complex, ndim=2, mode='c'] rho = np.zeros((nspf,)*2, dtype=complex)
    cdef tuple idxi,idxj
    for i in range(nspf):
        for j in range(nspf):
            idxi = (slice(None),)*mode + (i,...)
            idxj = (slice(None),)*mode + (j,...)
            rho[i,j] = np.sum(A[idxi].conj()*A[idxj])
    return rho

def invert_density_matrix(cnp.ndarray[complex, ndim=2, mode='c'] rho, regularization='default', eps=1.e-8):
    """Inverts given density matrix using standard regularization method.
    """
    if regularization=='default':
        try:
            w,v = np.linalg.eig(rho)
        except:
            print(rho)
            raise ValueError('rho was not diagonalized')
        w += eps*np.exp(-w/eps)
        w = 1./w
    return np.dot(v, np.dot(np.diag(w), v.conj().T))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _act_density(int nspf,int npbf,complex[:,::1] rho,complex[::1] spfs,complex[::1] spfsout) nogil:
    """Act density matrix on the single particle functions.
    """
    cdef int i,j
    cdef int ind_i,ind_j
    cdef int incx = 1
    cdef int incy = 1
    ind_i = 0
    for i in range(nspf):
        ind_j = 0
        for j in range(nspf):
            blas.zaxpy(&npbf,&rho[i,j],&spfs[ind_j:ind_j+npbf][0],&incx,&spfsout[ind_i:ind_i+npbf][0],&incy)
            ind_j += npbf
        ind_i += npbf
    return

def act_density(int nspf,int npbf,cnp.ndarray[complex, ndim=2, mode='c'] rho,cnp.ndarray[complex, ndim=1, mode='c'] spfs):
    """Act density matrix on the single particle functions.
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] spfsout = np.zeros_like(spfs, dtype=complex)
    _act_density(nspf,npbf,rho,spfs,spfsout)
    return spfsout

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_projector(int nspf,int npbf,complex[::1] spfs,complex[:,::1] proj) nogil:
    """
    """
    cdef int ind = 0
    cdef int incx = 1
    cdef int incy = 1
    cdef double alpha = 1.0
    for i in range(nspf):
        blas.zher('U',&npbf,&alpha,&spfs[ind:ind+npbf][0],&incx,&proj[0,0],&npbf)
        ind += npbf
    return

def compute_projector(int nspf,int npbf,cnp.ndarray[complex, ndim=1, mode='c']  spfs):
    cdef cnp.ndarray[complex, ndim=2, mode='c'] proj = np.zeros((npbf,npbf), dtype=complex)
    _compute_projector(nspf,npbf,spfs,proj)
    return proj

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _act_projector(int nspf,int npbf,complex[:,::1] proj,complex[::1] spfs,complex[::1] spfsout) nogil:
    """
    """
    cdef int i
    # act (1-proj) on spfs
    cdef int ind = 0
    cdef complex alpha = -1.0
    cdef complex beta  = 1.0
    cdef int incx = 1
    cdef int incy = 1
    for i in range(nspf):
        blas.zcopy(&npbf,&spfs[ind:ind+npbf][0],&incx,&spfsout[ind:ind+npbf][0],&incy)
        blas.zhemv('U',&npbf,&alpha,&proj[0,0],&npbf,&spfs[ind:ind+npbf][0],&incx,&beta,&spfsout[ind:ind+npbf][0],&incy)
        ind += npbf
    return

def act_projector(int nspf,int npbf,cnp.ndarray[complex, ndim=2, mode='c'] proj,cnp.ndarray[complex, ndim=1, mode='c'] spfs,cnp.ndarray[complex, ndim=1, mode='c'] spfsout):
    """
    """
    _act_projector(nspf,npbf,proj,spfs,spfsout)
    return

def overlap_matrices(int nel,int nmodes,cnp.ndarray[long, ndim=2, mode='c'] nspfs,cnp.ndarray[long, ndim=1, mode='c'] npbfs,cnp.ndarray[long, ndim=2, mode='c'] spfstart,cnp.ndarray[object, ndim=1, mode='c'] spfs):
    """Computes overlap matrices between spfs of a wavefunction
    """
    cdef int alpha,beta,mode
    cdef int ind0_l,indf_l,ind0_r,indf_r
    cdef list spfovs,spfovs_a,spfovs_b
    spfovs = []
    for alpha in range(nel-1):
        spfovs_a = []
        for beta in range(alpha+1,nel):
            spfovs_b = []
            for mode in range(nmodes):
                ind0_l = spfstart[alpha,mode]
                indf_l = ind0_l + nspfs[alpha,mode]*npbfs[mode]
                ind0_r = spfstart[beta,mode]
                indf_r = ind0_r + nspfs[beta,mode]*npbfs[mode]
                spfovs_b.append( spf_innerprod(nspfs[alpha,mode],nspfs[beta,mode],
                    npbfs[mode],spfs[alpha][ind0_l:indf_l],spfs[beta][ind0_r:indf_r]) )
            spfovs_a.append( spfovs_b )
        spfovs.append( spfovs_a )
    return spfovs

def overlap_matrices2(int nel,int nmodes,cnp.ndarray[int, ndim=2, mode='c'] nspfs,cnp.ndarray[int, ndim=1, mode='c'] npbfs,cnp.ndarray[int, ndim=2, mode='c'] spfstart,cnp.ndarray[object, ndim=2, mode='c'] spfs1,cnp.ndarray[object, ndim=2, mode='c'] spfs2):
    """Computes overlap matrices between spfs of two wavefunctions
    """
    cdef int alpha,beta,mode
    cdef int ind0_l,indf_l,ind0_r,indf_r
    cdef list spfovs,spfovs_a,spfovs_b
    spfovs = []
    for alpha in range(nel):
        spfovs_a = []
        for beta in range(alpha,nel):
            spfovs_b = []
            for mode in range(nmodes):
                ind0_l = spfstart[alpha,mode]
                indf_l = ind0_l + nspfs[alpha,mode]*npbfs[mode]
                ind0_r = spfstart[beta,mode]
                indf_r = ind0_r + nspfs[beta,mode]*npbfs[mode]
                spfovs_b.append( spf_innerprod(nspfs[alpha,mode],nspfs[beta,mode],
                    npbfs[mode],spfs1[alpha][ind0_l:indf_l],spfs2[beta][ind0_r:indf_r]) )
            spfovs_a.append( spfovs_b )
        spfovs.append( spfovs_a )
    return spfovs
