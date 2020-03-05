from copy import deepcopy

import numpy as np
import scipy.sparse as sp

import pymctdh.opfactory as opfactory

class PBasis:
    """
    """

    def __init__(self, args, combined=False, sparse=False):
        """
        """
        self.combined = combined
        self.sparse = sparse

        if self.combined:
            self.nmodes = len(args)
            # create new pbasis for each mode
            self.pbasis = []
            for i in range(self.nmodes):
                self.pbasis.append( PBasis(args[i], sparse=self.sparse) )
        else:
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
                #self.grid = make_ho_grid(self.params['npbf'])
            elif self.params['basis'] == 'sinc':
                self.params['npbf'] = args[1]
                self.params['qmin'] = args[2]
                self.params['qmax'] = args[3]
                self.params['dq']   = args[4]
                self.params['mass'] = args[5]
                if isinstance(self.params['npbf'], list):
                    self.make_ops = opfactory.make_sinc_ops_combined
                else:
                    self.make_ops = opfactory.make_sinc_ops
                self.grid = np.arange(qmin,qmax+dq,dq)
            elif self.params['basis'] == 'plane wave':
                if args[1]%2 == 0:
                    self.params['npbf'] = args[1]+1
                else:
                    self.params['npbf'] = args[1]
                self.params['nm']   = int((args[1]-1)/2)
                self.params['mass'] = args[2]
                if len(args) == 4:
                    self.combined = args[3]
                else:
                    self.combined = False
                if self.combined:
                    raise NotImplementedError
                else:
                    self.make_ops = opfactory.make_planewave_ops
            elif self.params['basis'] == 'plane wave dvr':
                raise NotImplementedError
                #if args[1]%2 == 0:
                #    self.params['npbf'] = args[1]+1
                #else:
                #    self.params['npbf'] = args[1]
                #self.params['nm']   = int((args[1]-1)/2)
                #self.params['mass'] = args[2]
                #if len(args) == 4:
                #    self.combined = args[3]
                #else:
                #    self.combined = False
                #if self.combined:
                #    raise NotImplementedError
                #else:
                #    self.make_ops = opfactory.make_planewave_ops
                #    #self.grid = np.arange(qmin,qmax+dq,dq)
            elif self.params['basis'] == 'radial':
                raise NotImplementedError
                #self.params['npbf'] = args[1]
                #self.params['dq']   = args[2]
                #self.params['mass'] = args[3]
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
            # TODO I think this was supposed to be for inputing custom operators
            #if matrix is None:
            #    matrix = [None for i in range(len(ops))]
            for i,op in enumerate(ops):
                if not op in self.ops:
                    self.ops[op] = self.make_ops(self.params,op,sparse=self.sparse)
                    #if matrix[i] is None:
                    #    self.ops[op] = self.make_ops(self.params,op,sparse=self.sparse)
                    #else:
                    #    self.ops[op] = matrix[i]
                ## TODO make this for custom operators
                #if isinstance(op,str):
                #    self.ops[op] = self.make_ops(params,op)
                #else:
                #    ind = 'c%d'%(count)
                #    count += 1
                #    self.ops[op] = op.copy()

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
        return

    def operate1b(self, spf, alpha):
        """Operate the single-body hamiltonian on a single spf.
        """
        return self.ops['1b'][alpha]@spf

    def operate(self, spf, term):
        """Operate a single-body term on a single spf.
        """
        return self.ops[term]@spf
