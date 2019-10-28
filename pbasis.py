import numpy as np
import opfactory
from cy.sparsemat import matvec

class PBasis:
    """
    """

    def __init__(self, args, sparse=False):
        """
        """
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
                #raise ValueError("I know I'm being difficult, but put in an odd number plz thank u")
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
                self.grid = np.arange(qmin,qmax+dq,dq)
        else:
            raise ValueError("Not a valid basis.")

    def make_operators(self, ops):
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
        for op in ops:
            if not op in self.ops:
                self.ops[op] = self.make_ops(self.params,op,sparse=self.sparse)
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
            return matvec(self.ops[term], spf)
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
