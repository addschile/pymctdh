import numpy as np
import pymctdh.linalg as LA

from .pbasis import PBasis
from .hamiltonian import Hamiltonian
from .cy.wftools import dvrtransform

from copy import deepcopy

class Wavefunction(object):
    """Wavefunction class that handles storage of wavefunction with which the 
    user can interact.
    """

    def __init__(self, nel, nmodes, nspfs, npbfs, A=None, spfs=None, 
                 noinds=False):
        """
        Attributes
        ----------
        nel - int, number of electronic states
        nmodes - int, number of modes (can be combined)
        nspfs - list or np.ndarray, number of single-particle functions in each
                mode
        npbfs - list or np.ndarray, number of primitive basis functions for each
                mode
        psi - np.ndarray(dtype=complex), 1-d array containing the mctdh core 
              tensor and single-particle functions
        npsi - int, size of psi array
        psistart - np.ndarray((2,nel),dtype=int), array containing initial
                   indices of the mctdh core tensor and single-particle 
                   functions for each electronic state
        psiend - np.ndarray((2,nel),dtype=int), array containing final indices
                 of the mctdh core tensor and single-particle functions for
                 each electronic state
        spfstart - np.ndarray((nel,nmodes),dtype=int), where the spfs for a 
                   mode start relative to the psistart index
        spfend - np.ndarray((nel,nmodes),dtype=int), where the spfs for a mode 
                 start relative to the psistart index
        combined - list of bools, is the pbasis for a mode is combined?
        cmodes - list of list of ints???

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
        self.info = {}
        # get number of electronic states and modes
        self.nel    = nel
        self.nmodes = nmodes
        self.info['nel'] = self.nel
        self.info['nmodes'] = self.nmodes
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
        self.info['nspfs'] = self.nspfs
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
        self.info['npbfs'] = self.npbfs

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
        self.info['npsi'] = self.npsi
        self.info['psistart'] = self.psistart
        self.info['psiend'] = self.psiend

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
            self.info['spfstart'] = self.spfstart
            self.info['spfend'] = self.spfend

    def copy(self,arg=None):
        """Copy the wavefunction and its parts
        """
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
        """Normalizes the coefficients of spfs, i.e., the core tensor
        """
        nrm = self.norm()
        ind = self.psiend[0,-1]
        self.psi[:ind] /= nrm

    def normalize_spfs(self):
        """Normalizes the spf vectors
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
        """Normalizes the wavefunction
        """
        # normalize coefficients
        self.normalize_coeffs()
        # normalize spf
        self.normalize_spfs()

    # TODO add to wftools for jit/cython
    def reshape_spfs(self, npbfs, nspfs, spfs):
        """Reshapes the given spfs from a vector into a matrix
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
        """Orthonormalizes the single-particle functions on each electronic 
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

    def transform_to_dvr(self, pbfs):
        """Wrapper function to function that transforms spfs to dvr basis.
        """
        self.psi = dvrtransform(self.nel,self.nmodes,self.npsi,self.nspfs,
                                self.npbfs,self.psistart,self.psiend,
                                self.spfstart,self.spfend,pbfs,self.psi)
