import numpy as np
from pbasis import PBasis
import linalg as LA
from copy import deepcopy
from hamiltonian import Hamiltonian

class Wavefunction(object):
    """
    """

    def __init__(self, nel, nmodes, nspfs, npbfs, A=None, spfs=None, 
                 noinds=False):
        """
        """
        # get number of electronic states and modes
        self.nel    = nel
        self.nmodes = nmodes
        # get number of spfs for each state and mode
        self.nspfs  = nspfs
        #if isinstance(nspfs, list)
        #    self.nspfs = np.zeros(nmodes, dtype=int)
        #    for i in range(len(nspfs)):
        #        if isinstance(nspfs[i], list):
        #            for j in range(len(nspfs[i])):
        #                if j==0:
        #                    self.nspfs[i] = int(nspfs[i][j])
        #                else:
        #                    self.nspfs[i] *= int(nspfs[i][j])
        #        else:
        #            self.nspfs[i] = int(nspfs[i])
        #else:
        #    self.nspfs = nspfs.astype(int)
        # get number of primitive basis functions for ith mode
        self.npbfs = npbfs.astype(int)

        if A==None:
            # generate lists with spf and A tensor arrays
            self.A    = np.zeros(self.nel, dtype=np.ndarray)
            for i in range(self.nel):
                Adim   = ()
                for j in range(self.nmodes):
                    Adim   += (self.nspfs[i,j],)
                self.A[i] = np.zeros(Adim, dtype=complex)
        else:
            self.A = deepcopy(A)

        if spfs==None:
            self.spfs = np.zeros(self.nel, dtype=np.ndarray)
            for i in range(self.nel):
                spfdim = 0
                for j in range(self.nmodes):
                    spfdim += self.nspfs[i,j]*self.npbfs[j]
                self.spfs[i] = np.zeros(spfdim, dtype=complex)
        else:
            self.spfs = deepcopy(spfs)

        # generate array that contains "pointer" to start of spf arrays for each mode
        if not noinds:
            self.spfstart = np.zeros((self.nel,self.nmodes), dtype=int)
            self.spfend   = np.zeros((self.nel,self.nmodes), dtype=int)
            for i in range(self.nel):
                ind = 0
                for j in range(self.nmodes):
                    self.spfstart[i,j] = ind
                    ind += self.nspfs[i,j]*self.npbfs[j]
                    self.spfend[i,j] = ind

    def copy(self,arg=None):
        if arg==None:
            out = Wavefunction(self.nel, self.nmodes, self.nspfs, self.npbfs, A=self.A, spfs=self.spfs, noinds=True)
            out.spfstart = self.spfstart
            out.spfend = self.spfend
        elif arg=='A':
            out = deepcopy(self.A)
        elif arg=='spfs':
            out = deepcopy(self.spfs)
        else:
            raise ValueError("Not a valid copying argument")
        return out

    def generate_ic(self, el0):
        """Creates basic initial conditions for wavefunction on el0-th
        electronic state.
        """
        for i in range(self.nel):
            for j in range(self.nmodes):
                nspf = self.nspfs[i,j]
                npbf = self.npbfs[j]
                count = 0
                ind = self.spfstart[i,j]
                for k in range(nspf):
                    self.spfs[i][ind+count] = 1.0
                    ind += npbf
                    count += 1
        self.A[el0][(0,)*self.nmodes] = 1.0

    def norm(self):
        """Computes norm of the wavefunction.
        """
        nrm = 0.0
        for i in range(self.nel):
            nrm += np.sum((self.A[i].conj()*self.A[i])).real
        return nrm

    def normalize_coeffs(self):
        """
        """
        nrm = self.norm()
        for alpha in range(self.nel):
            self.A[alpha] /= np.sqrt(nrm)

    def normalize_spfs(self):
        """
        """
        for i in range(self.nel):
            for j in range(self.nmodes):
                nspf = self.nspfs[i,j]
                npbf = self.npbfs[j]
                ind = self.spfstart[i,j]
                for k in range(nspf):
                    self.spfs[i][ind:ind+npbf] /= LA.norm(self.spfs[i][ind:ind+npbf])
                    ind += npbf
    
    def normalize(self):
        """
        """
        # normalize coefficients
        self.normalize_coeffs()
        # normalize spf
        self.normalize_spfs()

    # TODO add to wftools for jit/cython
    def reshape_spfs(self, npbfs, nspfs, spfs):
        """
        """
        if len(spfs.shape) == 1:
            # convert vector to matrix
            spfsout = np.zeros((npbfs,nspfs),dtype=complex)
            for i in range(nspfs):
                spfsout[:,i] = spfs[i*npbfs:(i+1)*npbfs]
        elif len(spfs.shape) == 2:
            # convert matrix to vector 
            spfsout = np.zeros(npbfs*nspfs,dtype=complex)
            for i in range(nspfs):
                spfsout[i*npbfs:(i+1)*npbfs] = spfs[:,i]
        else:
            raise ValueError("Single-particle functions have wrong shape")
        return spfsout

    def orthonormalize_spfs(self, spfsin=None, method='gram-schmidt'):
        """Orthonormalize the single-particle functions on each electronic 
        state.
        """
        if spfsin!=None:
            spfsout = deepcopy(spfsin)
        for i in range(self.nel):
            for j in range(self.nmodes):
                nspf = self.nspfs[i,j]
                npbf = self.npbfs[j]
                ind0 = self.spfstart[i,j]
                indf = self.spfend[i,j]
                if spfsin==None:
                    spfs = self.reshape_spfs(npbf,nspf,self.spfs[i][ind0:indf])
                    spfs = LA.orthonormalize(nspf, spfs, method=method)
                    self.spfs[i][ind0:indf] = self.reshape_spfs(npbf,nspf,spfs)
                else:
                    spfs = self.reshape_spfs(npbf,nspf,spfsin[i][ind0:indf])
                    spfs = LA.orthonormalize(nspf, spfs, method=method)
                    spfsout[i][ind0:indf] = self.reshape_spfs(npbf,nspf,spfs)
        if spfsin!=None:
            return spfsout
    
    # TODO need to change this so it's compatible with new stuff
    #def compute_energy(self, inp):
    #    """Computes energy of the wavefunction provided a Hamiltonian or dA/dt.
    #    """
    #    if isinstance(inp, Hamiltonian):
    #        # compute equation of motion and energy
    #        from optools import precompute_ops
    #        from eom import eom_coeffs
    #        # TODO change arguments
    #        opspfs, opips = precompute_ops(self.nel,self.nmodes,self.nspf,self.npbf,self.spfstart,self.spfend,ham.ops, self)
    #        self.overlap_matrices()
    #        dA = eom_coeffs(self, inp, opips)
    #        energy = 0.0
    #        for i in range(self.nel):
    #            energy += (1.j*np.sum(self.A[i].conj()*dA[i])).real
    #    elif isinstance(inp, Wavefunction):
    #        energy = 0.0
    #        for i in range(self.nel):
    #            energy += (1.j*np.sum(self.A[i].conj()*inp.A[i])).real
    #    return energy

    def diabatic_pops(self):
        """Computes diabatic populations for each electronic state.
        """
        pops = np.zeros(self.nel)
        for i in range(self.nel):
            pops[i] = np.sum(self.A[i].conj()*self.A[i]).real
        return pops

    # TODO
    #def avg_q(self, mode, nel=None):
    #    """Computes average position of a mode
    #    """
    #    if nel==None:
    #        # compute average on both surfaces
    #        for alpha in range(nel):
    #            Asum = tensorcontract([mode],wf.A[alpha,:])
    #        for i in range(self.nspf[mode]):
    #            
    #    else:
    #    pops = np.zeros(self.nel)
    #    for i in range(self.nel):
    #        pops[i] = np.sum(self.A[i,:].conj()*self.A[i,:]).real
    #    return pops

if __name__ == "__main__":

    nel    = 2
    nmodes = 4
    nspfs = np.array([[7, 12, 6, 5],
                     [7, 12, 6, 5]], dtype=int)
    npbfs = np.array([22, 32, 21, 12], dtype=int)
    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    wf.generate_ic(1)
    print(wf.A[1][(0,)*nmodes])
