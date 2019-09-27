import numpy as np

def norm2(v):
    """Compute norm squared of a vector.
    """
    return np.dot(v.conj(),v)

def norm(v):
    """Compute norm of a vector.
    """
    return np.sqrt(norm2(v))

def householder(v):
    """Computes the Househodler reflector.
    """
    raise NotImplementedError
    #u = v.copy()
    #nu = norm(v)
    #u /= nu
    #if u[0] >= 0.0:
    #    u[0] += 1.
    #    nu *= -1.
    #else:
    #    u[0] -= 1.
    #H = np.eye(len(u), dtype=u.dtype) - np.outer(u,u.conj())
    #return H

def orthogonalize(nvecs, vecs, method='gram-schmidt'):
    """Orthogonalizes a set of vectors.
    Inputs
    -----
    """
    vecsout = np.zeros_like(vecs, dtype=vecs.dtype)
    if method == 'householder':
        raise NotImplementedError
        #H = householder(vecs[:,0])
        #for i in range(nvecs):
        #    vecsout[:,i] = np.dot(H, vecs[:,i])
    elif method=='gram-schmidt':
        for i in range(nvecs):
            vecsout[:,i] = vecs[:,i] / norm(vecs[:,i])
            for j in range(i+1,nvecs):
                vecs[:,j] = vecs[:,j] - np.dot(vecsout[:,i].conj(),vecs[:,j])*vecsout[:,i]
    return vecsout

def orthonormalize(nvecs, vecs, method='gram-schmidt', normalize=False):
    """Orthonormalizes a set of vectors.
    """
    # orthogonalize
    vecsout = orthogonalize(nvecs, vecs, method=method)
    # normalize
    if normalize:
        for i in range(nvecs):
            vecsout[:,i] /= norm(vecsout[:,i])
    return vecsout
