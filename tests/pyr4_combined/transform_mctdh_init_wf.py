import numpy as np

def make_q(npbf):
    q = np.zeros((npbf,)*2)
    for i in range(npbf-1):
        q[i,i+1] = np.sqrt(0.5*float(i+1))
        q[i+1,i] = np.sqrt(0.5*float(i+1))
    return q

nel = 2
nmodes = 2
nspfs = np.array([[6,6],
                  [4,4]], dtype=int)
npbfs = [[12, 22],[12, 7]]
Adim = 0
for i in range(nel):
    Adim_tmp = 1
    for j in range(nmodes):
        Adim_tmp *= nspfs[i,j]
    Adim += Adim_tmp
spfdim = 0
for i in range(nel):
    for j in range(nmodes):
        spftmp = 1
        for k in range(len(npbfs[j])):
            spftmp *= npbfs[j][k]
        spfdim += spftmp*nspfs[i,j]

psi = np.zeros(Adim+spfdim,dtype=complex)
f = open('combined_psi.txt','r')
for i in range(len(psi)):
    line = f.readline().split()
    psi[i] = float(line[1])+1.j*float(line[3])
f.close()

psiout = np.zeros_like(psi,dtype=complex)
psiout[:Adim] = psi[:Adim]
psispfs = psi[Adim:]
ind = Adim
for i in range(nel):
    for j in range(nmodes):
        nspf = nspfs[i,j]
        npbf = npbfs[j][0]*npbfs[j][1]
        q1 = make_q(npbfs[j][0])
        q2 = make_q(npbfs[j][1])
        w,v1 = np.linalg.eigh(q1)
        w,v2 = np.linalg.eigh(q2)
        V1 = np.kron(v1,np.eye(npbfs[j][1]))
        V2 = np.kron(np.eye(npbfs[j][0]),v2)
        for k in range(nspf):
            spf = psi[ind:ind+npbf]
            psiout[ind:ind+npbf] = np.dot(V1,spf)
            psiout[ind:ind+npbf] = np.dot(V2,spf)
            ind += npbf
np.save('transformed_psi.npy',psiout)
np.savetxt('transformed_psi.txt',psiout)
