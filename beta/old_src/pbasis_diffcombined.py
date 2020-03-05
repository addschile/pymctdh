import numpy as np
import opfactory
#from cy.sparsemat import matvec
from numba import jit,njit

@njit(fastmath=True)
def matvec(nrows,IA,JA,data,vec,outvec):
    """Sparse matrix-vector multiplication for small(ish) matrices
    """
    d_ind = 0
    for i in range(nrows):
        ncol = IA[i+1]-IA[i]
        for j in range(ncol):
            col_ind = JA[d_ind]
            outvec[i] = outvec[i] + data[d_ind]*vec[col_ind]
            d_ind += 1
    return outvec

class PBasis:
    """General class that handles creation of operators in the primitive basis 
    and the application of operators on the single-particle functions. Each 
    instance of this class handles the primitive basis for a single-mode and
    can handle combined modes.
    """

    def __init__(self, args, combined=False, sparse=False):
        """
        Parameters
        ----------
        args - list (of lists), arguments that describes the primitive basis 
                (or primitive bases in a combined mode)
        combined - bool, whether or not a mode is a combined mode
        sparse - bool, use sparse matrix techniques or not
        """
        self.combined = combined
        if self.combined:
            self.params = list()
        self.params = {}
        self.params['basis'] = args[0].lower()
        self.sparse = sparse

        # set up parameters for basis
        if self.params['basis'] == 'ho':
            self.params['npbf']  = args[1]
            self.params['mass']  = args[2]
            self.params['omega'] = args[3]
            if len(args) == 5:
                self.combined = args[4]
            else:
                self.combined = False
            if self.combined:
                self.make_ops = opfactory.make_ho_ops_combined
                if not isinstance(self.params['mass'], list):
                    mlist = [args[2] for i in range(len(args[1]))]
                    self.params['mass'] = mlist
                if not isinstance(self.params['omega'], list):
                    omlist = [args[2] for i in range(len(args[1]))]
                    self.params['omega'] = omlist
            else:
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
        if matrix is None:
            matrix = [None for i in range(len(ops))]
        for i,op in enumerate(ops):
            if not op in self.ops:
                if matrix[i] is None:
                    self.ops[op] = self.make_ops(self.params,op,sparse=self.sparse)
                else:
                    self.ops[op] = matrix[i]
            ## TODO make this for custom operators
            #if isinstance(op,str):
            #    self.ops[op] = self.make_ops(params,op)
            #else:
            #    ind = 'c%d'%(count)
            #    count += 1
            #    self.ops[op] = op.copy()

    def operate(self, spf, term):
        """Operate a single-body term on a single spf.
        """
        #return self.ops[term]@spf
        if self.sparse:
            op = self.ops[term]
            outvec = np.zeros(op.nrows, dtype=complex)
            return matvec(op.nrows,op.IA,op.JA,op.data,spf,outvec)
            #return matvec(self.ops[term], spf)
        else:
            return np.dot(self.ops[term], spf)

if __name__ == "__main__":

    # no mode combination
    pbf = PBasis(['ho',22,1.0,1.0])
    pbf.make_operators(['q','KE','q^2'])
    print(pbf.params['basis'])
    print(pbf.params['npbf'])
    print(pbf.params['mass'])
    print(pbf.params['omega'])
    opkeys = pbf.ops.keys()
    for op in opkeys:
        print(op)
        print(pbf.ops[op].shape)
    print('')
    print('')

    # mode combination
    pbf = PBasis(['ho',[6,6],1.0,1.0,True])
    pbf.make_operators(['(q)*(1)','(1)*(q)'])
    print(pbf.params['basis'])
    print(pbf.params['npbf'])
    print(pbf.params['mass'])
    print(pbf.params['omega'])
    opkeys = pbf.ops.keys()
    for op in opkeys:
        print(op)
        print(pbf.ops[op].shape)
    print('')
    print('')

    # mode combination
    pbf = PBasis(['ho',[6,6],[1.0,2.0],[1.0,2.0],True])
    pbf.make_operators(['(q)*(1)','(1)*(q)'])
    print(pbf.params['basis'])
    print(pbf.params['npbf'])
    print(pbf.params['mass'])
    print(pbf.params['omega'])
    opkeys = pbf.ops.keys()
    for op in opkeys:
        print(op)
        print(pbf.ops[op].shape)
    print('')
    print('')
