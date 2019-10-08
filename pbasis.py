import numpy as np
import opfactory

# TODO add mode combination here
class PBasis:
    """
    """

    def __init__(self, args):
        """
        """
        self.params = {}
        self.params['basis'] = args[0].lower()

        # set up parameters for basis
        if self.params['basis'] == 'ho':
            self.params['npbf']  = args[1]
            self.params['mass']  = args[2]
            self.params['omega'] = args[3]
            if isinstance(self.params['npbf'], list):
                self.make_ops = opfactory.make_ho_ops_combined
                if not isinstance(self.params['mass'], list):
                    mlist = [args[2] for i in range(len(args[1]))]
                    self.params['mass'] = mlist
                if not isinstance(self.params['omega'], list):
                    omlist = [args[2] for i in range(len(args[1]))]
                    self.params['omega'] = omlist
            else:
                self.make_ops = opfactory.make_ho_ops
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
        elif self.basis == 'plane wave' or self.basis == 'fft':
            self.params['npbf'] = args[1]
            # TODO
            self.params['kmin'] = args[2]
            self.params['kmax'] = args[3]
            self.params['dk']   = args[4]
            self.params['mass'] = args[5]
        #elif self.basis == 'custom':
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
        #count = 0
        for op in ops:
            if not op in self.ops:
                self.ops[op] = self.make_ops(self.params,op)
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
        if '^' in term:
            spf_tmp = spf.copy()
            for i in range(int(term[-1])):
                spf_tmp = self.operate(spf_tmp, term[0])
            return spf_tmp
        elif '*' in term:
            _term = term.split('*')
            spf_tmp = spf.copy()
            for i in range(len(_term)):
                spf_tmp = self.operate(spf_tmp, _term[i])
            return spf_tmp
        return np.dot(self.ops[term], spf)

if __name__ == "__main__":

    # no mode combination
    pbf = PBasis(['ho',22,1.0,1.0])
    pbf.make_operators(['q','KE','q^2'])
    #opkeys = pbf.ops.keys()
    #for op in opkeys:
    #    print(op)
    #    print(pbf.ops[op])
    #    print('')
    #    print('')

    # mode combination
    pbf = PBasis(['ho',[6,6,6,6],1.0,1.0])
    #pbf.make_operators(['q*1*1*1','KE*1*1*1','1*q*1*1','1*KE*1*1'])
    pbf.make_operators(['q*1*1*1','1*q*1*1'])
    opkeys = pbf.ops.keys()
    eye = np.eye(6)
    q = np.zeros((6,)*2)
    for i in range(6-1):
        q[i,i+1] = np.sqrt(0.5*float(i+1))
        q[i+1,i] = np.sqrt(0.5*float(i+1))
    for i,op in enumerate(opkeys):
        print(op)
        print(pbf.ops[op].shape)
        if i==0:
            oper = np.kron(q,np.kron(eye,np.kron(eye,eye)))
            print(np.allclose(oper,pbf.ops[op]))
        else:
            oper = np.kron(eye,np.kron(q,np.kron(eye,eye)))
            print(np.allclose(oper,pbf.ops[op]))
        print('')
        print('')

    nel    = 2
    nmodes = 7
    nspfs = np.array([[19, 13, 10, 8, 6, 6, 5],
                      [11,  8,  6, 4, 4, 4, 4]], dtype=int)
    npbfs = [[22,32],[21,12],[6,6,6,6],[6,6,6,6],[6,6,6,6],[6,6,6,6],[6,6,6,6]]
    pbfs = list()
    for i in range(len(npbfs)):
        pbfs.append( PBasis(['ho', npbfs[i], 1.0, 1.0]) )
