import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numba import jit,double,int64

@jit(double(double,double))
def cos2phi(qc,qt):
    e1=3.94
    e2=4.84
    k1t=-0.105
    k2t=0.149
    lamda=0.262
    Delta = 0.5*(e2+k2t*qt-e1-k1t*qt)
    return Delta / np.sqrt( Delta**2. + lamda**2.*qc**2. )

@jit(double(double,double))
def sin2phi(qc,qt):
    e1=3.94
    e2=4.84
    k1t=-0.105
    k2t=0.149
    lamda=0.262
    Delta = 0.5*(e2+k2t*qt-e1-k1t*qt)
    return lamda*qc / np.sqrt( Delta**2. + lamda**2.*qc**2. )

@jit()
def make_kmat(nc,nt,alpha1,alpha2,qc,qt,gam):
    print(nc,nt)
    Kmat = np.zeros((nc*nt,nc*nt))
    nmax = nc*nt
    for i in range(nc):
        for k in range(nt):
            ind1 = i*nt + k
            for ip in range(nc):
                for kp in range(nt):
                    ind2 = ip*nt + kp
                    Kmat[ind1,ind2] = np.exp((-alpha1*(qc[i]-qc[ip])**2.) + (-alpha2*2.*(np.sin(0.5*(qt[k] - qt[kp])))**2.))
                    if ind1 == ind2:
                        Kmat[ind1,ind1] += gam**2.
    return Kmat

@jit()
def make_vkrr(nc,nt,ncd,ntd,qc,qt,qcd,qtd,alpha1,alpha2,w1,w2,w12):
    vkrr1  = np.zeros((ncd,ntd))
    vkrr2  = np.zeros((ncd,ntd))
    vkrr12 = np.zeros((ncd,ntd))
    for i in range(ncd):
        for j in range(ntd):
            for ip in range(nc):
                for jp in range(nt):
                    k = ip*nt + jp
                    vkrr1[i,j]  +=  w1[k]*np.exp((-alpha1*(qcd[i]-qc[ip])**2.) + (-alpha2*2.*(np.sin(0.5*(qtd[j] - qt[jp])))**2.))
                    vkrr2[i,j]  +=  w2[k]*np.exp((-alpha1*(qcd[i]-qc[ip])**2.) + (-alpha2*2.*(np.sin(0.5*(qtd[j] - qt[jp])))**2.))
                    vkrr12[i,j] += w12[k]*np.exp((-alpha1*(qcd[i]-qc[ip])**2.) + (-alpha2*2.*(np.sin(0.5*(qtd[j] - qt[jp])))**2.))
    return vkrr1,vkrr2,vkrr12

wc     = 0.19
kappa0 = 0.0
kappa1 = 0.095
minv   = 1.43e-3
E0     = 0.0
E1     = 2.00
W0     = 2.3
W1     = 1.50
lamda  = 0.19

# parameters for kernel matrix
alpha1 = 0.1
alpha2 = 0.1
gam = 0.0001

# grid for potential
qc  = np.arange(-6.,6.,0.5)
nc  = len(qc)
qt = np.arange(-0.5*np.pi,0.5*3*np.pi,0.1*np.pi)
nt = len(qt)

# diabatic potentials
v1 = np.zeros((nc,nt))
v2 = np.zeros((nc,nt))
v12 = np.zeros((nc,nt))
for i in range(nc):
    for j in range(nt):
        v1[i,j]  = E0 + 0.5*W0 + 0.5*wc*qc[i]**2. - 0.5*W0*np.cos(qt[j])
        v2[i,j]  = E1 - 0.5*W1 + 0.5*wc*qc[i]**2. + kappa1*qc[i] + 0.5*W1*np.cos(qt[j])
        v12[i,j] = lamda*qc[i]

print('making kmat')
Kmat = make_kmat(nc,nt,alpha1,alpha2,qc,qt,gam)
print('inverting kmat')
#Kmat = make_kmat(nc,nt,alpha,qc,qt,gam)
Kinv = np.linalg.inv(Kmat)

w1  = np.dot(Kinv,v1.flatten())
w2  = np.dot(Kinv,v2.flatten())
w12 = np.dot(Kinv,v12.flatten())

print('plotting diabatic states')
# grid for potential
qcd = np.arange(-6.,6.,0.01)
ncd = len(qcd)
qtd = np.arange(-0.5*np.pi,0.5*3*np.pi,0.01)
ntd = len(qtd)

# diabatic potentials
v1  = np.zeros((ncd,ntd))
v2  = np.zeros((ncd,ntd))
v12 = np.zeros((ncd,ntd))
for i in range(ncd):
    for j in range(ntd):
        v1[i,j]  = E0 + 0.5*W0 + 0.5*wc*qcd[i]**2. - 0.5*W0*np.cos(qtd[j])
        v2[i,j]  = E1 - 0.5*W1 + 0.5*wc*qcd[i]**2. + kappa1*qcd[i] + 0.5*W1*np.cos(qtd[j])
        v12[i,j] = lamda*qcd[i]
vkrr1,vkrr2,vkrr12 = make_vkrr(nc,nt,ncd,ntd,qc,qt,qcd,qtd,alpha1,alpha2,w1,w2,w12)

QC,QT = np.meshgrid(qtd,qcd)
# diabatic
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(QC,QT,v1,alpha=0.5)
ax.plot_surface(QC,QT,v2,alpha=0.5)
ax.set_title('diabatic')
ax.set_xlabel(r'$q_c$')
ax.set_ylabel(r'$q_t$')
plt.tight_layout()
#plt.savefig('og_db_pes.png')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(QC,QT,vkrr1,alpha=0.5)
ax.plot_surface(QC,QT,vkrr2,alpha=0.5)
ax.set_title('diabatic krr')
ax.set_xlabel(r'$q_c$')
ax.set_ylabel(r'$q_t$')
plt.tight_layout()
#plt.savefig('krr_db_pes.png')
plt.show()

plt.contourf(QC,QT,vkrr1-v1)
plt.xlabel(r'$q_c$')
plt.ylabel(r'$q_t$')
plt.tight_layout()
#plt.savefig('krr_db_pes.png')
plt.colorbar()
plt.show()
plt.contourf(QC,QT,vkrr2-v2)
plt.xlabel(r'$q_c$')
plt.ylabel(r'$q_t$')
plt.tight_layout()
plt.colorbar()
#plt.savefig('krr_db_pes.png')
plt.show()
