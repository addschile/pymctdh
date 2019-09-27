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
            self.make_ops = opfactory.make_ho_ops
        elif self.params['basis'] == 'sinc':
            self.params['npbf'] = args[1]
            self.params['qmin'] = args[2]
            self.params['qmax'] = args[3]
            self.params['dq']   = args[4]
            self.params['mass'] = args[5]
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

    pbf = PBasis(['ho',22,1.0,1.0])
