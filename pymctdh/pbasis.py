from copy import deepcopy

import numpy as np
import scipy.sparse as sp

import pymctdh.opfactory as opfactory

class PBasis:
    """
    """

    def __init__(self, args, combined=False, sparse=False, makegrid=True):
        """
        """
        self.combined = combined
        self.sparse = sparse
        self.dvrproj = None

        if self.combined:
            self.sparse = True
            self.nmodes = len(args)
            # create new pbasis for each mode
            self.pbasis = []
            for i in range(self.nmodes):
                self.pbasis.append( PBasis(args[i], sparse=self.sparse) )
        else:
            self.nmodes = 1
            self.params = {}
            self.params['basis'] = args[0].lower()
            self.sparse = sparse

            # set up parameters for basis
            if self.params['basis'] == 'ho':
                self.params['npbf']  = args[1]
                self.params['mass']  = args[2]
                self.params['omega'] = args[3]
                self.npbfs = self.params['npbf']
                self.make_ops = opfactory.make_ho_ops
                if makegrid:
                    if self.combined:
                        self.vgrid = []
                        for i in range(self.nmodes):
                            vgrid = sp.lil_matrix(self.pbasis[i].vgrid)
                            for j in range(self.nmodes):
                                if i<j:
                                    vgrid = sp.kron(self.vgrid, sp.eye(self.pbasis.params['npbf'], fmt='lil'))
                                elif i>j:
                                    vgrid = sp.kron(sp.eye(self.pbasis.params['npbf'], fmt='lil'), self.vgrid)
                            self.vgrid.append( sp.csr_matrix(vgrid) )
                    else:
                        self.grid,self.vgrid = opfactory.make_ho_grid(self.params)
                try:
                    self.params['dvr'] = args[4]
                except:
                    self.params['dvr'] = False
            elif self.params['basis'] == 'sinc':
                self.params['npbf'] = args[1]
                self.params['qmin'] = args[2]
                self.params['qmax'] = args[3]
                self.params['dq']   = args[4]
                self.params['mass'] = args[5]
                self.params['dvr'] = True
                self.make_ops = opfactory.make_sinc_ops
                self.grid = np.arange(qmin,qmax+dq,dq)
                self.vgrid = None
            elif self.params['basis'] == 'plane wave':
                if args[1]%2 == 0:
                    self.params['npbf'] = args[1]+1
                else:
                    self.params['npbf'] = args[1]
                self.params['nm']   = int((args[1]-1)/2)
                self.params['mass'] = args[2]
                self.make_ops = opfactory.make_planewave_ops
                try:
                    self.params['dvr'] = args[4]
                except:
                    self.params['dvr'] = False
                if self.params['dvr']:
                    raise NotImplementedError
            elif self.params['basis'] == 'plane wave dvr':
                raise NotImplementedError
            elif self.params['basis'] == 'radial':
                raise NotImplementedError
            else:
                raise ValueError("Not a valid basis.")

    def make_operators(self, ops, matrix=None):
        """Creates matrices for all the relevant operators used in the 
        calculation. These matrices are then stored in a dictionary called
        self.ops.

        Input
        -----
        ops - list of strings, all the operators that are used for this pbf
        """
        try:
            self.ops
        except:
            self.ops = {}
        if self.combined:
            for op in ops:
                if '(' and ')' in op:
                    op_ = op.split(')*(')
                    if len(op_) != self.nmodes:
                        print('Number of modes in term: %d'%(len(op_)))
                        print('Number of modes in combined mode: %d'%(self.nmodes))
                        raise ValueError("Incorrect number of terms for this combined mode.")
                    for i in range(len(op_)):
                        op_[i] = op_[i].split('(')[-1]
                        op_[i] = op_[i].split(')')[0]
                        self.pbasis[i].make_operators([op_[i]])
                        if i==0:
                            opmat = self.pbasis[i].ops[op_[i]]
                        else:
                            if self.sparse:
                                opmat = sp.kron(opmat,self.pbasis[i].ops[op_[i]])
                            else:
                                opmat = np.kron(opmat,self.pbasis[i].ops[op_[i]])
                    self.ops[op] = opmat
                elif op == '1':
                    op_ = ['1' for i in range(self.nmodes)]
                    for i in range(len(op_)):
                        self.pbasis[i].make_operators([op_[i]])
                        if i==0:
                            opmat = self.pbasis[i].ops[op_[i]]
                        else:
                            if self.sparse:
                                opmat = sp.kron(opmat,self.pbasis[i].ops[op_[i]])
                            else:
                                opmat = np.kron(opmat,self.pbasis[i].ops[op_[i]])
                    self.ops[op] = opmat
                else:
                    raise ValueError("Incorrectly defined term for combined mode.")
        else:
            for i,op in enumerate(ops):
                if not op in self.ops:
                    self.ops[op] = self.make_ops(self.params,op,sparse=self.sparse)
                    if self.params['dvr']:
                        if not self.vgrid is None:
                            self.ops[op] = self.vgrid.conj().T@self.ops[op]@self.vgrid
                            #print(op)
                            #print(np.diag(self.ops[op]))
                            #print('')

    def make_1b_ham(self, nel, terms):
        """Make the 1-body hamiltonians that act on the spfs with this pbf.
        """
        op1b = []
        for alpha in range(nel):
            op = None
            for term in terms[alpha]:
                opstr = term['ops'][0]
                coeff = term['coeff']
                if type(coeff) == np.complex:
                    if not op is None:
                        op = op.astype('complex')
                if op is None:
                    op = coeff*self.ops[opstr]
                else:
                    op += coeff*self.ops[opstr]
            op1b.append( op )
        self.ops['1b'] = op1b

    def operate1b(self, spf, alpha):
        """Operate the single-body hamiltonian on a single spf.
        """
        return self.ops['1b'][alpha]@spf

    def operate(self, spf, term):
        """Operate a single-body term on a single spf.
        """
        return self.ops[term]@spf

    def dvrtrans(self, spf):
        """Returns the DVR transformed
        """
        #if self.params['dvr']:
        #    # the spfs are already in the dvr grid basis
        #    return spf
        #else:
        return self.vgrid.conj().T@spf

    def makedvrproj(self):
        """Make projection operators onto grid points of dvr basis.
        """
        if self.combined:
            for i in range(self.nmodes):
                self.pbasis[i].makedvrproj()
            self.dvrproj = list()
            for i in range(self.nmodes):
                modeproj = list()
                for j in range(self.pbasis.params['npbf']):
                    proj = self.pbasis.dvrproj[j]
                    for k in range(self.nmodes):
                        if k<i:
                            proj = sp.kron(sp.eye((self.pbasis[k].params['npbf'],)*2,format='csr'),proj)
                        elif k>i:
                            proj = sp.kron(proj,sp.eye((self.pbasis[k].params['npbf'],)*2,format='csr'))
                    modeproj.append( proj )
                self.dvrproj.append( modeproj )
        else:
            self.dvrproj = list()
            for i in range(self.params['npbf']):
                self.dvrproj.append( sp.csr_matrix(([1.], ([i],[i])), shape=(self.params['npbf'],)*2) )
