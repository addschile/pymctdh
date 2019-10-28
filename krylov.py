import numpy as np
from scipy.linalg import expm
from optools import matel
from cy.wftools import norm,inner
import units as units

def lanczos(A, nel, nmodes, nspfs, npbfs, ham, uopips, copips, spfovs, nvecs=5, 
            return_evecs=True):
    """
    """
    # thing to store A tensors
    V = np.zeros(nvecs, np.ndarray)
    V[0] = A/norm(nel,A)
    T = np.zeros((nvecs,nvecs))

    # form krylov vectors and tridiagonal matrix
    for i in range(nvecs-1):
        np.set_printoptions(threshold=np.inf)
        #print(V[i])
        V[i+1] = matel(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                       ham.hcelterms,spfovs,V[i])
        #print(V[i+1])
        #raise ValueError
        # compute alpha 
        T[i,i] = inner(nel,V[i],V[i+1]).real
        if i>0:
            V[i+1] += -T[i,i]*V[i] - T[i-1,i]*V[i-1]
        else:
            V[i+1] += -T[i,i]*V[i]
        # normalize previous vector
        nvprev = norm(nel,V[i+1])
        V[i+1] /= nvprev
        # compute beta
        T[i,i+1] = nvprev
        T[i+1,i] = T[i,i+1]
    Vtmp = matel(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                 ham.hcelterms,spfovs,V[-1])
    T[-1,-1] = inner(nel,V[-1], Vtmp).real

    if return_evecs:
        return T , V
    else:
        return T

def arnoldi(A, nel, nmodes, nspfs, npbfs, ham, uopips, copips, spfovs, nvecs=5, 
            return_evecs=True):
    """
    """
    # thing to store A tensors
    V = np.zeros(nvecs, np.ndarray)
    V[0] = A/norm(nel,A)

    # form krylov subspace and upper hessenberg matrix
    T = np.zeros((nvecs,nvecs), dtype=complex)
    for j in range(nvecs-1):
        w = matel(nel,nmodes,nspfs,npbfs,uopips,copips,ham.huelterms,
                       ham.hcelterms,spfovs,V[i])
        for i in range(j+1):
            T[i,j] = inner(nel, w, V[i])
            w -= T[i,j]*V[i]
        if j < nvecs-1:
            T[j+1,j] = norm(nel,w)
            V[j+1] = w/T[j+1,j]

    return T , V

################################################################################
# Functions that use krylov subspace methods for creating all the vectors      #
################################################################################
def propagate(V, T, dt):
    """
    """
    nvecs = len(V)
    psiprop = expm(-1.j*dt*T)[:,0]
    #print(psiprop[-1])
    for i in range(nvecs):
        if i==0:
            psiout = psiprop[i]*V[i]
        else:
            psiout += psiprop[i]*V[i]
    return psiout

def krylov_prop(y, dt, nel, nmodes, nspfs, npbfs, ham, uopips, copips, spfovs,
                method='lanczos', return_all=False):

    if method == 'arnoldi':
        T , V = arnoldi(y,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,nvecs=5, 
                        return_evecs=True)
    else:
        T , V = lanczos(y,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,nvecs=10,
                        return_evecs=True)

    #for i in range(len(V)-1):
    #    if i==0:
    #        err = np.abs(T[i,i+1]*dt)
    #    else:
    #        err = np.abs(err*T[i,i+1]*dt/float(i+1))
    #print(err)

    # do the propagation
    y = propagate(V, T, dt)

    return y

def krylov_prop_ada(t_start, t_finish, dt, y, nel, nmodes, nspfs, npbfs, ham, uopips, copips, spfovs,
                method='lanczos', return_all=False):

    h = dt
    t0 = t_start
    while abs(t0 - t_finish) > 1.e-12:
        #print(t0,h)
        # change timestep if too large
        if h > (t_finish-t0):
            h = t_finish-t0
    
        if method == 'arnoldi':
            T , V = arnoldi(y,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,nvecs=5, 
                            return_evecs=True)
        else:
            T , V = lanczos(y,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,nvecs=10,
                            return_evecs=True)

        for i in range(len(V)-1):
            if i==0:
                err = np.abs(T[i,i+1]*h)
            else:
                err = np.abs(err*T[i,i+1]*h/float(i+1))

        if err > 1.e-10:
        #if False:#err > 1.e-10:
            #print(err,h)
            h *= (1.e-10/err)**(1./float(len(V)))
            #for i in range(len(V)-1):
            #    if i==0:
            #        err = np.abs(T[i,i+1]*h)
            #    else:
            #        err = np.abs(err*T[i,i+1]*h/float(i+1))
            #print(err,h)

        # do the propagation
        y = propagate(V, T, dt)

        #print(np.diag(T))
        #print(np.diagonal(T,offset=1))
        #raise ValueError
        #nvecs = len(V)
        #w,v = np.linalg.eigh(T)
        #print(w)
        #T , V = lanczos(y,nel,nmodes,nspfs,npbfs,ham,uopips,copips,spfovs,nvecs=11,
        #                return_evecs=True)
        #w,v = np.linalg.eigh(T)
        #print(w)
        ##print(np.exp(-1.j*w*h))
        ##print(np.dot(v,np.dot(np.diag(np.exp(-1.j*w*h)),v.conj().T))[:,0])
        ##print(expm(-1.j*dt*T)[:,0])
        #raise ValueError

        #print(np.abs(w[0])*h)
        #print(np.exp(-1.j*w[0]*h))
        #if np.exp(-1.j*w[0]*h).real < 0.999:
        #    print('hey',h,np.exp(-1.j*w[0]*h).real)
        ##if np.abs(w[0])*h < 0.999:
        #    # adjust timestep
        #    #h = -np.log(0.999)/np.abs(w[0])
        #    h = -np.log(0.999)/np.abs(w[0])
        #    #h = 1.e-12/np.abs(w[0])
        #    print(h)

        ## make propagator
        #psiprop = expm(-1.j*h*T)[:,0]
        #if np.abs(psiprop[-1]) > 1.e-12:
        #    h *= 0.5
        #    psiprop = expm(-1.j*h*T)[:,0]

        ## do the propagation
        #for i in range(nvecs):
        #    if i==0:
        #        y = psiprop[i]*V[i]
        #    else:
        #        y += psiprop[i]*V[i]

        # adjust time
        t0 += h

    return y
