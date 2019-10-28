import numpy as np

def make_q(npbf):
    q = np.zeros((npbf,)*2)
    for i in range(npbf-1):
        q[i,i+1] = np.sqrt(0.5*float(i+1))
        q[i+1,i] = np.sqrt(0.5*float(i+1))
    return q

nel = 2
nmodes = 4
nspfs = np.array([[7, 12, 6, 5],
                  [7, 12, 6, 5]], dtype=int)
npbfs = np.array([22, 32, 21, 12], dtype=int)
psi = np.load('init_wf.npy')
psiout = np.zeros_like(psi,dtype=complex)
Adim = 0
for i in range(nel):
    Adim_tmp = 1
    for j in range(nmodes):
        Adim_tmp *= nspfs[i,j]
    Adim += Adim_tmp
psiout[:Adim] = psi[:Adim]
psispfs = psi[Adim:]
ind = Adim
for i in range(nel):
    for j in range(nmodes):
        nspf = nspfs[i,j]
        npbf = npbfs[j]
        q = make_q(npbfs[j])
        w,v = np.linalg.eigh(q)
        for k in range(nspf):
            spf = psi[ind:ind+npbf]
            psiout[ind:ind+npbf] = np.dot(v.conj().T,spf)
            ind += npbf
np.save('transformed_psi.npy',psiout)
np.savetxt('transformed_psi.txt',psiout)
