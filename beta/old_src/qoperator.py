import numpy as np

import units

class QOperator(object):
    """Generic operator class for user defined potentials.
    """

    def __init__(self, nmodes, term, pbfs=None):
        """
        """
        self.nmodes = nmodes
        self.term = term
        self.term_setup()
        self.op_setup()
        if pbfs != None:
            # make all the operators for the pbf
            for i in range(nmodes):
                pbfs[i].make_operators(self._ops[i])

    def term_setup(self):
        """Setup function that makes sure all keywords are properly defined and
           defaulted.
        """
        # check coefficient
        if not 'coeff' in self.term:
            self.term['coeff'] = 1.0
        # check units
        if not 'units' in self.term:
            self.term['units'] = 'au'
        # do unit conversions for the coefficients
        self.term['units'] = self.term['units'].lower()
        self.term['coeff'] *= units.convert_to(self.term['units'])
        # default electronic operator to 1
        if not 'elop' in self.term:
            self.term['elop'] = '1'
        if 'modes' in self.term:
            # if there are modes make sure it's a list
            if not isinstance(self.term['modes'], list):
                self.term['modes'] = [self.term['modes']]
            if not isinstance(self.term['ops'], list):
                self.term['ops'] = [self.term['ops']]

    def op_setup(self):
        """
        """
        self.ops = [[] for i in range(self.nmodes)]
        if 'modes' in self.term: # check that it's not purely electronic
            modes = self.term['modes']
            ops = self.term['ops']
            for i in range(len(modes)):
                self.ops[modes[i]].append( ops[i] )
        self._ops = [[] for i in range(self.nmodes)]
        for i in range(self.nmodes):
            for op in self.ops[i]:
                # TODO this will not work for something like p*q^2, but whatever
                if '^' in op:
                    _op = op[0]
                elif '*' in op:
                    _op = op.split('*')
                else:
                    _op = op.split()
                for j in range(len(_op)):
                    if len(self._ops[i]) == 0:
                        self._ops[i].append( _op[j] )
                    else:
                        if not _op[j] in self._ops[i]:
                            self._ops[i].append( _op[j] )

if __name__ == "__main__":

    nmodes = 4
    term = {'coeff': 0.01, 'units': 'ev', 'modes': 0, 'ops': 'q'}

    op = QOperator(nmodes, term)

    print('ops')
    print(op.ops)
    print('primitive ops')
    print(op._ops)

    term = {'coeff': 0.01, 'units': 'ev', 'modes': 0, 'ops': 'q^2'}
    op = QOperator(nmodes, term)
    print('ops')
    print(op.ops)
    print('primitive ops')
    print(op._ops)

    term = {'coeff': 0.01, 'units': 'ev', 'modes': 0, 'ops': 'q*q'}
    op = QOperator(nmodes, term)
    print('ops')
    print(op.ops)
    print('primitive ops')
    print(op._ops)
