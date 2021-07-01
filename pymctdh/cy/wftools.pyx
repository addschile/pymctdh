import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas

def norm2(int nel,cnp.ndarray A):
    """
    """
    cdef double nrm = 0.0
    for i in range(nel):
        nrm += np.sum(A[i].conj()*A[i]).real
    return nrm

def norm(int nel,cnp.ndarray A):
    cdef double nrm = norm2(nel,A)
    return np.sqrt(nrm)

def inner(int nel,cnp.ndarray A1,cnp.ndarray A2):
    cdef complex innr = 0.0j
    for i in range(nel):
        innr += np.sum(A1[i].conj()*A2[i])
    return innr

# TODO
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void _spf_innerprod_par(int nspf_l,int nspf_r,int npbf,complex[::1] spfs1,complex[::1] spfs2,complex[:,::1] spfsout) nogil:
#    """
#    """
#    cdef int i,j
#    cdef int ind_i,ind_j
#    cdef int incx = 1
#    cdef int incy = 1
#    ind_i = 0
#    for i in range(nspf_l):
#        ind_j = 0
#        for j in range(nspf_r):
#            spfsout[i,j] = blas.zdotc(&npbf,&spfs1[ind_i],&incx,&spfs2[ind_j],&incy)
#            ind_j += npbf
#        ind_i += npbf
#    return

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
            spfsout[i,j] = blas.zdotc(&npbf,&spfs1[ind_i],&incx,&spfs2[ind_j],&incy)
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

# TODO
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void _multAtens_par(int nspf_l,int nspf_r,int npbf,complex[:,::1] A,complex[::1] spfs,complex[::1] spfsout) nogil:
#    """Computes the multiplication of the A tensor with the single-particle 
#       functions.
#    """
#    cdef int i,j
#    cdef int ind_i,ind_j
#    cdef int incx = 1
#    cdef int incy = 1
#    ind_i = 0
#    for i in range(nspf_l):
#        ind_j = 0
#        for j in range(nspf_r):
#            blas.zaxpy(&npbf,&A[i,j],&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
#            ind_j += npbf
#        ind_i += npbf
#    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _multAtens(int nspf_l,int nspf_r,int npbf,complex[:,::1] A,complex[::1] spfs,complex[::1] spfsout) nogil:
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
            blas.zaxpy(&npbf,&A[i,j],&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
            ind_j += npbf
        ind_i += npbf
    return

def multAtens(int nspf_l,int nspf_r,int npbf,cnp.ndarray[complex, ndim=2, mode='c'] A,cnp.ndarray[complex, ndim=1, mode='c'] spfs):
    """Computes the multiplication of the A tensor with the single-particle 
       functions.
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] spfsout = np.zeros(nspf_l*npbf, dtype=complex)
    _multAtens(nspf_l,nspf_r,npbf,A,spfs,spfsout)
    return spfsout

def compute_density_matrix(int nspf,int alpha,int nmodes,int mode,cnp.ndarray A):
    """Computes density matrix for given mode and electronic states.
    """
    cdef int i
    cdef cnp.ndarray[complex, ndim=2, mode='c'] rho = np.zeros((nspf,)*2, dtype=complex)
    cdef list modes = [i for i in range(nmodes)]
    modes.remove( mode )
    return np.tensordot(A.conj(),A,axes=[modes,modes])

def invert_density_matrix(cnp.ndarray[complex, ndim=2, mode='c'] rho, regularization='default', eps=1.e-8):
    """Inverts given density matrix using standard regularization method.
    """
    if regularization=='default':
        try:
            w,v = np.linalg.eigh(rho)
        except:
            print(rho)
            raise ValueError('rho was not diagonalized')
        w += eps*np.exp(-abs(w)/eps)
        w = 1./w
    return np.dot(v, np.dot(np.diag(w), v.conj().T))

# TODO
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void _act_density_par(int nspf,int npbf,complex[:,::1] rho,complex[::1] spfs,complex[::1] spfsout) nogil:
#    """Act density matrix on the single particle functions.
#    """
#    cdef int i,j
#    cdef int ind_i,ind_j
#    cdef int incx = 1
#    cdef int incy = 1
#    ind_i = 0
#    for i in range(nspf):
#        ind_j = 0
#        for j in range(nspf):
#            blas.zaxpy(&npbf,&rho[i,j],&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
#            ind_j += npbf
#        ind_i += npbf
#    return

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
            blas.zaxpy(&npbf,&rho[i,j],&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
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
        blas.zher('U',&npbf,&alpha,&spfs[ind],&incx,&proj[0,0],&npbf)
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
        blas.zcopy(&npbf,&spfsout[ind],&incx,&spfs[0],&incy)
        blas.zhemv('U',&npbf,&alpha,&proj[0,0],&npbf,&spfs[0],&incx,&beta,&spfsout[ind],&incy)
        ind += npbf
    return

def act_projector(int nspf,int npbf,cnp.ndarray[complex, ndim=2, mode='c'] proj,cnp.ndarray[complex, ndim=1, mode='c'] spfsout):
    """
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] spfs = np.zeros(npbf, dtype=complex)
    _act_projector(nspf,npbf,proj,spfs,spfsout)
    return

# TODO
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef void _project_par(int nspf,int npbf,complex[::1] spf_tmp,complex[::1] spfs,complex[::1] spfsout) nogil:
#    """
#    """
#    cdef int i,j,k
#    cdef int ind_i,ind_j
#    cdef int incx = 1
#    cdef int incy = 1
#    cdef complex inp = 0.0
#    cdef complex mone = -1.0
#    cdef complex zero = 0.0
#    ind_i = 0
#    for i in range(nspf):
#        ind_j = 0
#        for j in range(nspf):
#            inp = blas.zdotc(&npbf,&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
#            blas.zaxpy(&npbf,&inp,&spfs[ind_j],&incx,&spf_tmp[0],&incy)
#            ind_j += npbf
#        blas.zaxpy(&npbf,&mone,&spf_tmp[0],&incx,&spfsout[ind_i],&incy)
#        blas.zscal(&npbf,&zero,&spf_tmp[0],&incx)
#        ind_i += npbf
#    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _project(int nspf,int npbf,complex[::1] spf_tmp,complex[::1] spfs,complex[::1] spfsout) nogil:
    """
    """
    cdef int i,j,k
    cdef int ind_i,ind_j
    cdef int incx = 1
    cdef int incy = 1
    cdef complex inp = 0.0
    cdef complex mone = -1.0
    cdef complex zero = 0.0
    ind_i = 0
    for i in range(nspf):
        ind_j = 0
        for j in range(nspf):
            inp = blas.zdotc(&npbf,&spfs[ind_j],&incx,&spfsout[ind_i],&incy)
            blas.zaxpy(&npbf,&inp,&spfs[ind_j],&incx,&spf_tmp[0],&incy)
            ind_j += npbf
        blas.zaxpy(&npbf,&mone,&spf_tmp[0],&incx,&spfsout[ind_i],&incy)
        blas.zscal(&npbf,&zero,&spf_tmp[0],&incx)
        ind_i += npbf
    return

def project(int nspf,int npbf,cnp.ndarray[complex, ndim=1, mode='c'] spfs,cnp.ndarray[complex, ndim=1, mode='c'] spfsout):
    """
    """
    cdef cnp.ndarray[complex, ndim=1, mode='c'] spf_tmp = np.zeros(npbf, dtype=complex)
    _project(nspf,npbf,spf_tmp,spfs,spfsout)
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

def overlap_matrices2(int nel,int nmodes,cnp.ndarray[long, ndim=2, mode='c'] nspfs,cnp.ndarray[long, ndim=1, mode='c'] npbfs,cnp.ndarray[long, ndim=2, mode='c'] spfstart,cnp.ndarray[object, ndim=1, mode='c'] spfs1,cnp.ndarray[object, ndim=1, mode='c'] spfs2):
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

def norm(cnp.ndarray[long, ndim=2, mode='c'] psiend, cnp.ndarray[complex, ndim=1, mode='c'] psi):
    """Computes norm of the wavefunction.
    """
    cdef int ind = psiend[0,-1]
    cdef double nrm = np.sum((psi[:ind].conj()*psi[:ind])).real
    return np.sqrt(nrm)

def normalize_coeffs(cnp.ndarray[long, ndim=2, mode='c'] psiend, cnp.ndarray[complex, ndim=1, mode='c'] psi):
    """Normalizes the coefficients of spfs, i.e., the core tensor
    """
    cdef double nrm = norm(psiend,psi)
    cdef int ind = psiend[0,-1]
    psi[:ind] /= nrm
    return

def normalize_spfs(int nel,int nmodes,cnp.ndarray[long, ndim=2, mode='c'] nspfs,cnp.ndarray[long, ndim=1, mode='c'] npbfs,cnp.ndarray[long, ndim=2, mode='c'] spfstart,cnp.ndarray[long, ndim=2, mode='c'] psistart,cnp.ndarray[long, ndim=2, mode='c'] psiend,cnp.ndarray[complex, ndim=1, mode='c'] psi):
    """Normalizes the spf vectors
    """
    cdef int i,j
    cdef int ind,nspf,npbf
    cdef double nrm
    for i in range(nel):
        ind = psistart[1,i]
        for j in range(nmodes):
            nspf = nspfs[i,j]
            npbf = npbfs[j]
            for k in range(nspf):
                nrm = np.sqrt(np.dot(psi[ind:ind+npbf].conj(),psi[ind:ind+npbf]).real)
                if abs(nrm) > 1.e-30:
                    psi[ind:ind+npbf] /= nrm
                ind += npbf
    return

def normalize_wf(int nel,int nmodes,cnp.ndarray[long, ndim=2, mode='c'] nspfs,cnp.ndarray[long, ndim=1, mode='c'] npbfs,cnp.ndarray[long, ndim=2, mode='c'] spfstart,cnp.ndarray[long, ndim=2, mode='c'] psistart,cnp.ndarray[long, ndim=2, mode='c'] psiend,cnp.ndarray[complex, ndim=1, mode='c'] psi):
    """Normalizes the wavefunction
    """
    # normalize coefficients
    normalize_coeffs(psiend,psi)
    # normalize spf
    normalize_spfs(nel,nmodes,nspfs,npbfs,spfstart,psistart,psiend,psi)
    return

def reshape_wf(int nel,int nmodes,cnp.ndarray[long, ndim=2, mode='c'] nspfs,cnp.ndarray[long, ndim=2, mode='c'] psistart,cnp.ndarray[long, ndim=2, mode='c'] psiend,cnp.ndarray[complex, ndim=1, mode='c'] psi):
    """Reshapes one-dimensional wavefunction array into core tensor and spfs.
    """
    cdef int alpha,mode
    cdef tuple shaper
    # reshpae y into A tensor and spfs
    cdef cnp.ndarray A = np.zeros(nel, dtype=np.ndarray)
    cdef cnp.ndarray spfs = np.zeros(nel, dtype=np.ndarray)
    for alpha in range(nel):
        shaper = ()
        for mode in range(nmodes):
            shaper += (nspfs[alpha,mode],)
        # set A
        ind0 = psistart[0,alpha]
        indf = psiend[0,alpha]
        A[alpha] = np.reshape(psi[ind0:indf], shaper, order='C')
        # set spfs
        ind0 = psistart[1,alpha]
        indf = psiend[1,alpha]
        spfs[alpha] = psi[ind0:indf]
    return A,spfs

def reshape_wf_back(int nel,int npsi,cnp.ndarray[long, ndim=2, mode='c'] psistart,cnp.ndarray[long, ndim=2, mode='c'] psiend,cnp.ndarray A,cnp.ndarray spfs):
    """Reshapes core tensor and spfs into one-dimensional wavefunction array.
    """
    cdef int alpha,ind0,indf
    cdef cnp.ndarray[complex, ndim=1, mode='c'] psi = np.zeros(npsi, dtype=complex)
    for alpha in range(nel):
        ind0 = psistart[0,alpha]
        indf = psiend[0,alpha]
        psi[ind0:indf] = A[alpha].ravel()
        ind0 = psistart[1,alpha]
        indf = psiend[1,alpha]
        psi[ind0:indf] = spfs[alpha]
    return psi

def dvrtransform(nel,nmodes,npsi,nspfs,npbfs,psistart,psiend,spfstart,spfend,
                       pbfs,psi):
    """Function that computes populations at each grid point in diabatic states.
    """

    # reshape wf to core tensor and spfs
    A,spfs = reshape_wf(nel,nmodes,nspfs,psistart,psiend,psi)

    for alpha in range(nel):
        spf = spfs[alpha]
        for mode in range(nmodes):
            pbf = pbfs[mode]
            if pbf.combined == True:
                raise NotImplementedError
            else:
                npbf = pbf.params['npbf']
                ind0 = spfstart[alpha,mode]
                indf = spfend[alpha,mode]
                # transform spfs for this mode to dvr basis
                for i in range(nspfs[alpha,mode]):
                    indf = ind0 + npbf
                    spf[ind0:indf] = pbf.dvrtrans(spf[ind0:indf])
                    ind0 += npbf

    # reshape back to 1d array
    return reshape_wf_back(nel,npsi,psistart,psiend,A,spfs)
