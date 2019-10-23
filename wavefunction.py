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
        Attributes
        ----------
        nel
        nmodes
        nspfs
        npbfs
        combined
        cmodes
        npsi
        psistart
        psiend
        psi
        spfstart
        spfend

        Methods
        -------
        copy
        generate_ic
        norm
        normalize_coeffs
        normalize_spfs
        normalize
        reshape_spfs
        orthonormalize_spfs
        diabatic_pops
        """
        # get number of electronic states and modes
        self.nel    = nel
        self.nmodes = nmodes
        # get number of spfs for each state and mode
        if isinstance(nspfs, list):
            self.nspfs = np.zeros((nel,nmodes), dtype=int)
            for i in range(len(nspfs)):
                if isinstance(nspfs[i], list):
                    for j in range(len(nspfs[i])):
                        self.nspfs[i,j] = int(nspfs[i][j])
                else:
                    self.nspfs[0,i] = int(nspfs[i])
        else:
            self.nspfs = nspfs.astype(int)
        # get number of primitive basis functions for ith mode
        if isinstance(npbfs, list):
            self.npbfs = np.zeros(self.nmodes, dtype=int)
            self.combined = []
            self.cmodes = []
            for i in range(self.nmodes):
                if isinstance(npbfs[i], list):
                    if len(npbfs[i]) > 1:
                        self.combined.append( True )
                        cmodes = []
                        for j in range(len(npbfs[i])):
                            cmodes.append( npbfs[i][j] )
                            if j==0:
                                self.npbfs[i] = npbfs[i][j]
                            else:
                                self.npbfs[i] *= npbfs[i][j]
                        self.cmodes.append( cmodes )
                    else:
                        self.combined.append( False )
                        self.npbfs[i] = npbfs[i][0]
                else:
                    self.combined.append( False )
                    self.npbfs[i] = npbfs[i]
                    self.cmodes.append( npbfs[i] )
        else:
            self.npbfs = npbfs.astype(int)
            self.combined = [False for i in range(nmodes)]

        # initialize size of wavefunction array
        self.npsi     = 0
        self.psistart = np.zeros((2,nel), dtype=int)
        self.psiend   = np.zeros((2,nel), dtype=int)
        # start with A tensor
        for i in range(self.nel):
            self.psistart[0,i] = self.npsi
            ind = 1
            for j in range(self.nmodes):
                ind *= self.nspfs[i,j]
            self.npsi += ind
            self.psiend[0,i] = self.npsi
        # now add spfs
        for i in range(self.nel):
            self.psistart[1,i] = self.npsi
            for j in range(self.nmodes):
                nspf = self.nspfs[i,j]
                npbf = self.npbfs[j]
                self.npsi += nspf*npbf
            self.psiend[1,i] = self.npsi
        self.psi = np.zeros(self.npsi, dtype=complex)

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
            out = Wavefunction(self.nel, self.nmodes, self.nspfs, self.npbfs, noinds=True)
            out.spfstart = self.spfstart
            out.spfend = self.spfend
            out.psi = self.psi.copy()
        else:
            raise ValueError("Not a valid copying argument")
        return out

    def generate_ic(self, el0):
        """Creates basic initial conditions for wavefunction on el0-th
        electronic state.
        """
        for i in range(self.nel):
            ind = self.psistart[1,i]
            for j in range(self.nmodes):
                if self.combined[j]:
                    nmodes = len(self.cmodes[j])
                    cmodes = self.cmodes[j]
                    nspf   = self.nspfs[i,j]
                    npbf   = self.npbfs[j]
                    # make total ground state first
                    for l in range(nmodes):
                        spf_tmp = np.zeros(cmodes[l], dtype=complex)
                        spf_tmp[0] = 1.
                        if l==0:
                            spf_ = spf_tmp.copy()
                        else:
                            spf_ = np.kron(spf_,spf_tmp)
                    self.psi[ind:ind+npbf] += spf_
                    ind += npbf
                    excount = 1
                    modecount = 0
                    for k in range(nspf-1):
                        for l in range(nmodes):
                            if l==modecount:
                                spf_tmp = np.zeros(cmodes[l],dtype=complex)
                                spf_tmp[excount] = 1.0
                            else:
                                spf_tmp = np.zeros(cmodes[l],dtype=complex)
                                spf_tmp[0] = 1.0
                            if l==0:
                                spf_ = spf_tmp.copy()
                            else:
                                spf_ = np.kron(spf_,spf_tmp)
                        modecount += 1
                        if modecount == nmodes:
                            modecount = 0
                            excount += 1
                        self.psi[ind:ind+npbf] += spf_
                        ind += npbf
                else:
                    nspf = self.nspfs[i,j]
                    npbf = self.npbfs[j]
                    count = 0
                    for k in range(nspf):
                        self.psi[ind+count] = 1.0
                        ind += npbf
                        count += 1
        ind0 = self.psistart[0,el0]
        self.psi[ind0] = 1.0

    def norm(self):
        """Computes norm of the wavefunction.
        """
        ind = self.psiend[0,-1]
        nrm = np.sum((self.psi[:ind].conj()*self.psi[:ind])).real
        return np.sqrt(nrm)

    def normalize_coeffs(self):
        """
        """
        nrm = self.norm()
        ind = self.psiend[0,-1]
        self.psi[:ind] /= nrm

    def normalize_spfs(self):
        """
        """
        for i in range(self.nel):
            ind = self.psistart[1,i]
            for j in range(self.nmodes):
                nspf = self.nspfs[i,j]
                npbf = self.npbfs[j]
                for k in range(nspf):
                    nrm = LA.norm(self.psi[ind:ind+npbf])
                    if abs(nrm) > 1.e-30:
                        self.psi[ind:ind+npbf] /= nrm
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

    # TODO this needs testing
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
                    spfs = self.reshape_spfs(npbf,nspf,spfsin[i][ind0:indf])
                    spfs = LA.orthonormalize(nspf, spfs, method=method)
                    spfsout[i][ind0:indf] = self.reshape_spfs(npbf,nspf,spfs)
            return spfsout
        else:
            for i in range(self.nel):
                ind = self.psistart[1,i]
                for j in range(self.nmodes):
                    ind0 = self.spfstart[i,j]
                    indf = self.spfend[i,j]
                    nspf = self.nspfs[i,j]
                    npbf = self.npbfs[j]
                    spfs = self.reshape_spfs(npbf,nspf,self.psi[ind+ind0:ind+indf])
                    spfs = LA.orthonormalize(nspf, spfs, method=method)
                    self.psi[ind+ind0:ind+indf] = self.reshape_spfs(npbf,nspf,spfs)
    
    def diabatic_pops(self):
        """Computes diabatic populations for each electronic state.
        """
        pops = np.zeros(self.nel)
        for i in range(self.nel):
            ind0 = self.psistart[0,i]
            indf = self.psiend[0,i]
            pops[i] = np.sum(self.psi[ind0:indf].conj()*self.psi[ind0:indf]).real
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

    ### 4-mode pyrazine model ###
    print('4-mode pyrazine model')
    nel    = 2
    nmodes = 4
    nspfs = np.array([[7, 12, 6, 5],
                     [7, 12, 6, 5]], dtype=int)
    npbfs = np.array([22, 32, 21, 12], dtype=int)
    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    print(wf.combined)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')

    ### 4-mode pyrazine model list input ###
    print('4-mode pyrazine model list input')
    nel    = 2
    nmodes = 4
    nspfs = [[7, 12, 6, 5],[7, 12, 6, 5]]
    npbfs = [22, 32, 21, 12]
    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    print(wf.combined)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')

    ### 4-mode pyrazine model with combined modes ###
    print('4-mode pyrazine model with 2 combined modes')
    nel    = 2
    nmodes = 2
    nspfs = np.array([[8, 8],[7, 7]], dtype=int)
    npbfs = [[17, 27],[17, 10]]
    wf = Wavefunction(nel, nmodes, nspfs, npbfs)
    ind0 = wf.psistart[1,0]
    indf = wf.psiend[1,0]
    print(wf.psi[ind0:indf].shape)
    ind0 = wf.psistart[1,1]
    indf = wf.psiend[1,1]
    print(wf.psi[ind0:indf].shape)
    print(wf.combined)
    print(wf.cmodes)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    print(wf.spfstart)
    print(wf.spfend)
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')
