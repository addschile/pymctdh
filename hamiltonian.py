import numpy as np
import units
from optools import isdiag

class Hamiltonian(object):

    def __init__(self, nmodes, hterms, pbfs=None):
        """
        """
        # number of modes
        self.nmodes = nmodes
        # number of terms in the Hamiltonian
        self.nterms = len(hterms)
        # list housing all the terms in the hamiltonian
        self.hterms = hterms
        self.hterms_setup()
        # make list of all unique operators that act on spfs
        self.compute_unique_ops()
        # reform hterms lists to do correlated and uncorrelated terms
        self.correlate()
        if pbfs != None:
            # make all the operators for the pbf
            for i in range(nmodes):
                pbfs[i].make_operators(self._ops[i])

    def hterms_setup(self):
        """Setup function that makes sure all keywords are properly defined and
           defaulted.
        """
        for i in range(self.nterms):
            # check coefficient
            if not 'coeff' in self.hterms[i]:
                self.hterms[i]['coeff'] = 1.0
            # check units
            if not 'units' in self.hterms[i]:
                self.hterms[i]['units'] = 'au'
            # do unit conversions for the coefficients
            self.hterms[i]['units'] = self.hterms[i]['units'].lower()
            self.hterms[i]['coeff'] *= units.convert_to(self.hterms[i]['units'])
            # default electronic operator to 1
            if not 'elop' in self.hterms[i]:
                self.hterms[i]['elop'] = '1'
            if 'modes' in self.hterms[i]:
                # if there are modes make sure it's a list
                if not isinstance(self.hterms[i]['modes'], list):
                    self.hterms[i]['modes'] = [self.hterms[i]['modes']]
                if not isinstance(self.hterms[i]['ops'], list):
                    self.hterms[i]['ops'] = [self.hterms[i]['ops']]

    def correlate(self):
        """Splits the terms into correlated and uncorrelated pieces.
        """
        self.hcterms = []
        self.huterms = []
        for hterm in self.hterms:
            # uncorrelated terms must be electronically diagonal
            if isdiag(hterm['elop']):
                if 'modes' in hterm:
                    if len(hterm['modes'])==1:
                        self.huterms.append( hterm )
                    else:
                        self.hcterms.append( hterm )
                else:
                    self.huterms.append( hterm )
            else:
                self.hcterms.append( hterm )

    def compute_unique_ops(self):
        """Makes list of all unique operators that act on spfs
        """
        self.ops = [[] for i in range(self.nmodes)]
        for hterm in self.hterms:
            if 'modes' in hterm: # check that it's not purely electronic
                modes = hterm['modes']
                ops = hterm['ops']
                for i in range(len(modes)):
                    if len(self.ops[modes[i]]) == 0:
                        self.ops[modes[i]].append( ops[i] )
                    else:
                        if not ops[i] in self.ops[modes[i]]:
                            self.ops[modes[i]].append( ops[i] )
        self._ops = [[] for i in range(self.nmodes)]
        for i in range(self.nmodes):
            for op in self.ops[i]:
                if '^' in op:
                    _op = op.split('^')
                    _op = [_op[0]]
                #elif '*' in op:
                #    _op = op.split('*')
                else:
                    _op = op.split()
                for j in range(len(_op)):
                    if len(self._ops[i]) == 0:
                        self._ops[i].append( _op[j] )
                    else:
                        if not _op[j] in self._ops[i]:
                            self._ops[i].append( _op[j] )

if __name__ == "__main__":

    nmodes =  4
    w10a   =  0.09357
    w6a    =  0.0740 
    w1     =  0.1273 
    w9a    =  0.1568 
    delta  =  0.46165
    lamda  =  0.1825 
    k6a1   = -0.0964 
    k6a2   =  0.1194 
    k11    =  0.0470 
    k12    =  0.2012 
    k9a1   =  0.1594 
    k9a2   =  0.0484 

    hterms = []
    hterms.append({'coeff':   -delta, 'units': 'ev', 'elop': 'sz'}) # el only operator
    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  'KE'}) # mode 1 terms
    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': 'q^2'})
    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 1, 'ops':  'KE'}) # mode 2 terms
    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 1, 'ops': 'q^2'})
    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 2, 'ops':  'KE'}) # mode 3 terms
    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 2, 'ops': 'q^2'})
    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 3, 'ops':  'KE'}) # mode 4 terms
    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 3, 'ops': 'q^2'})
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': 'q'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 2 el 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 2, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 2, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 3, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 3, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 4 el 1

    ham = Hamiltonian(nmodes, hterms)
    print('ops')
    print(ham.ops)
    print('correlated terms')
    print(ham.hcterms)
    print('uncorrelated terms')
    print(ham.huterms)


########################################################################
####           Linear 24-mode VC model for Pyrazine
####  4-mode model of Domcke et al +  20-mode oscillator bath
####   Worth, Meyer and Cederbaum. J.Chem.Phys. 109 (1998) 3518
########################################################################
#
#OP_DEFINE-SECTION
#title
#Pyrazine 24-mode model with linear bath.
#end-title
#end-op_define-section
#
#PARAMETER-SECTION
#w10a    = 0.09357 , ev
#w6a   = 0.0740  , ev
#w1    = 0.1273  , ev
#w9a   = 0.1568  , ev
#w1b   = 0.0400  ,ev
#w2b   = 0.0589  ,ev
#w3b   = 0.0778  ,ev
#w4b   = 0.0968  ,ev
#w5b   = 0.1157  ,ev
#w6b   = 0.1347  ,ev
#w7b   = 0.1536  ,ev
#w8b   = 0.1726  ,ev
#w9b   = 0.1915  ,ev
#w10b   = 0.2105  ,ev
#w11b   = 0.2294  ,ev
#w12b   = 0.2484  ,ev
#w13b   = 0.2673  ,ev
#w14b   = 0.2863  ,ev
#w15b   = 0.3052  ,ev
#w16b   = 0.3242  ,ev
#w17b   = 0.3431  ,ev
#w18b   = 0.3621  ,ev
#w19b   = 0.3810  ,ev
#w20b   = 0.4000  ,ev
#
#delta = 0.46165 , ev
#lambda = 0.1825 , ev
#
#k6a1   = -0.0964 , ev     k6a2  = 0.1194 , ev
#k11    = 0.0470 , ev      k12   = 0.2012 , ev
#k9a1   = 0.1594 , ev      k9a2   = 0.0484 , ev
#k1b1   = 0.0069 , ev      k1b2   = -0.0069 , ev
#k2b1   = 0.0112 , ev      k2b2   = -0.0112 , ev
#k3b1   = 0.0102 , ev      k3b2   = -0.0102 , ev
#k4b1   = 0.0188 , ev      k4b2   = -0.0188 , ev
#k5b1   = 0.0261 , ev      k5b2   = -0.0261 , ev
#k6b1   = 0.0308 , ev      k6b2   = -0.0308 , ev
#k7b1   = 0.0210 , ev      k7b2   = -0.0210 , ev
#k8b1   = 0.0265 , ev      k8b2   = -0.0265 , ev
#k9b1   = 0.0196 , ev      k9b2   = -0.0196 , ev
#k10b1   = 0.0281 , ev     k10b2   = -0.0281 , ev
#k11b1   = 0.0284 , ev     k11b2   = -0.0284 , ev
#k12b1   = 0.0361 , ev     k12b2   = -0.0361 , ev
#k13b1   = 0.0560 , ev     k13b2   = -0.0560 , ev
#k14b1   = 0.0433 , ev     k14b2   = -0.0433 , ev
#k15b1   = 0.0625 , ev     k15b2   = -0.0625 , ev
#k16b1   = 0.0717 , ev     k16b2   = -0.0717 , ev
#k17b1   = 0.0782 , ev     k17b2   = -0.0782 , ev
#k18b1   = 0.0780 , ev     k18b2   = -0.0780 , ev
#k19b1   = 0.0269 , ev     k19b2   = -0.0269 , ev
#k20b1   = 0.0306 , ev     k20b2   = -0.0306 , ev
#end-parameter-section
#
#    hterms = []
#    hterms.append({'coeff':   -delta, 'units': 'ev', 'elop': 'sz'}) # el only operator
#    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  'KE'}) # mode 1 terms
#    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': 'q^2'})
#    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 1, 'ops':  'KE'}) # mode 2 terms
#    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 1, 'ops': 'q^2'})
#    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 2, 'ops':  'KE'}) # mode 3 terms
#    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 2, 'ops': 'q^2'})
#    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 3, 'ops':  'KE'}) # mode 4 terms
#    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 3, 'ops': 'q^2'})
#    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': 'q'}) # Peierls copuling
#    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 2 el 0
#    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 2 el 1
#    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 2, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 3 el 0
#    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 2, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 3 el 1
#    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 3, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 4 el 0
#    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 3, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 4 el 1
#
#HAMILTONIAN-SECTION
#modes  |  el  | v10a  | v6a  | v1  | v9a
#modes  |  1b  | 2b    | 3b   | 4b  | 5b   | 6b  | 7b  | 8b  | 9b  | 10b
#modes  | 11b  | 12b   | 13b  | 14b | 15b  | 16b | 17b | 18b | 19b | 20b
#
#sbasis-section
#multi-set
#      v10a, v6a         = 19, 11
#      v1, v9a           = 13, 8
#      6b, 17b, 5b, 16b  = 10, 6
#      18b, 13b, 15b, 4b =  8, 6
#      2b, 1b, 8b, 14b   =  6, 4
#      12b, 7b, 10b, 3b  =  6, 4
#      11b, 9b, 20b, 19b =  5, 4
#end-sbasis-section
#
#    ### electronic energy gap ###
#    hterms.append({'coeff': -delta, 'units': 'ev', 'elop': 'sz'})
#    ### mode 0 combined modes 10a and 6a ###
#    # diag terms
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'ops': 'KE*1'})
#    hterms.append({'coeff': w6a, 'units': 'ev', 'modes': 0, 'ops': '1*KE'})
#    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': 'q^2*1'})
#    hterms.append({'coeff': 0.5*w6a, 'units': 'ev', 'modes': 0, 'ops': '1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q'})
#    # peierls coupling
#    hterms.append({'coeff': lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': '1*q'})
#    ### mode 1 combined modes 1 and 9a ###
#    hterms.append({'coeff': w1, 'units': 'ev', 'modes': 1, 'ops': 'KE*1'})
#    hterms.append({'coeff': w9a, 'units': 'ev', 'modes': 1, 'ops': '1*KE'})
#    hterms.append({'coeff': 0.5*w1, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1'})
#    hterms.append({'coeff': 0.5*w9a, 'units': 'ev', 'modes': 1, 'ops': '1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q'})
#    ### mode 2 combined modes 6b, 17b, 5b, 16b ###
#    hterms.append({'coeff': w6b, 'units': 'ev', 'modes': 1, 'ops': 'KE*1*1*1'})
#    hterms.append({'coeff': 0.5*w6b, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1*1*1'})
#    hterms.append({'coeff': w17b, 'units': 'ev', 'modes': 1, 'ops': '1*KE*1*1'})
#    hterms.append({'coeff': 0.5*w17b, 'units': 'ev', 'modes': 1, 'ops': '1*q^2*1*1'})
#    hterms.append({'coeff': w5b, 'units': 'ev', 'modes': 1, 'ops': '1*1*KE*1'})
#    hterms.append({'coeff': 0.5*w5b, 'units': 'ev', 'modes': 1, 'ops': '1*1*q^2*1'})
#    hterms.append({'coeff': w16b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*KE'})
#    hterms.append({'coeff': 0.5*w16b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*1*q'})
#    ### mode 3 combined modes 18b, 13b, 15b, 4b ###
#    hterms.append({'coeff': w18b, 'units': 'ev', 'modes': 1, 'ops': 'KE*1*1*1'})
#    hterms.append({'coeff': 0.5*w18b, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1*1*1'})
#    hterms.append({'coeff': w13b, 'units': 'ev', 'modes': 1, 'ops': '1*KE*1*1'})
#    hterms.append({'coeff': 0.5*w13b, 'units': 'ev', 'modes': 1, 'ops': '1*q^2*1*1'})
#    hterms.append({'coeff': w15b, 'units': 'ev', 'modes': 1, 'ops': '1*1*KE*1'})
#    hterms.append({'coeff': 0.5*w15b, 'units': 'ev', 'modes': 1, 'ops': '1*1*q^2*1'})
#    hterms.append({'coeff': w4b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*KE'})
#    hterms.append({'coeff': 0.5*w4b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*1*q'})
#    ### mode 4 combined modes 2b, 1b, 8b, 14b ###
#    hterms.append({'coeff': w2b 'units': 'ev', 'modes': 1, 'ops': 'KE*1*1*1'})
#    hterms.append({'coeff': 0.5*w2b, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1*1*1'})
#    hterms.append({'coeff': w1b, 'units': 'ev', 'modes': 1, 'ops': '1*KE*1*1'})
#    hterms.append({'coeff': 0.5*w1b, 'units': 'ev', 'modes': 1, 'ops': '1*q^2*1*1'})
#    hterms.append({'coeff': w8b, 'units': 'ev', 'modes': 1, 'ops': '1*1*KE*1'})
#    hterms.append({'coeff': 0.5*w8b, 'units': 'ev', 'modes': 1, 'ops': '1*1*q^2*1'})
#    hterms.append({'coeff': w14b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*KE'})
#    hterms.append({'coeff': 0.5*w14b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*1*q'})
#    ### mode 5 combined modes 12b, 7b, 10b, 3b ###
#    hterms.append({'coeff': w12b, 'units': 'ev', 'modes': 1, 'ops': 'KE*1*1*1'})
#    hterms.append({'coeff': 0.5*w12b, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1*1*1'})
#    hterms.append({'coeff': w7b, 'units': 'ev', 'modes': 1, 'ops': '1*KE*1*1'})
#    hterms.append({'coeff': 0.5*w7b, 'units': 'ev', 'modes': 1, 'ops': '1*q^2*1*1'})
#    hterms.append({'coeff': w10b, 'units': 'ev', 'modes': 1, 'ops': '1*1*KE*1'})
#    hterms.append({'coeff': 0.5*w10b, 'units': 'ev', 'modes': 1, 'ops': '1*1*q^2*1'})
#    hterms.append({'coeff': w3b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*KE'})
#    hterms.append({'coeff': 0.5*w3b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*1*q'})
#    ### mode 6 combined modes 11b, 9b, 20b, 19b ###
#    hterms.append({'coeff': w11b, 'units': 'ev', 'modes': 1, 'ops': 'KE*1*1*1'})
#    hterms.append({'coeff': 0.5*w11b, 'units': 'ev', 'modes': 1, 'ops': 'q^2*1*1*1'})
#    hterms.append({'coeff': w9b, 'units': 'ev', 'modes': 1, 'ops': '1*KE*1*1'})
#    hterms.append({'coeff': 0.5*w9b, 'units': 'ev', 'modes': 1, 'ops': '1*q^2*1*1'})
#    hterms.append({'coeff': w20b, 'units': 'ev', 'modes': 1, 'ops': '1*1*KE*1'})
#    hterms.append({'coeff': 0.5*w20b, 'units': 'ev', 'modes': 1, 'ops': '1*1*q^2*1'})
#    hterms.append({'coeff': w19b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*KE'})
#    hterms.append({'coeff': 0.5*w19b, 'units': 'ev', 'modes': 1, 'ops': '1*1*1*q^2'})
#    # holstein couplings
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': 'q*1*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*q*1*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*q*1'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '1*1*1*q'})
#    hterms.append({'coeff': w10a, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '1*1*1*q'})
#
#
#
#k6a1            |1 S1&1  |3 q
#k6a2            |1 S2&2  |3 q
#k11             |1 S1&1  |4 q
#k12             |1 S2&2  |4 q
#k9a1            |1 S1&1  |5 q
#k9a2            |1 S2&2  |5 q
#k1b1            |1 S1&1  |6 q
#k1b2            |1 S2&2  |6 q
#k2b1            |1 S1&1  |7 q
#k2b2            |1 S2&2  |7 q
#k3b1            |1 S1&1  |8 q
#k3b2            |1 S2&2  |8 q
#k4b1            |1 S1&1  |9 q
#k4b2            |1 S2&2  |9 q
#k5b1            |1 S1&1  |10 q
#k5b2            |1 S2&2  |10 q
#k6b1            |1 S1&1  |11 q
#k6b2            |1 S2&2  |11 q
#k7b1            |1 S1&1  |12 q
#k7b2            |1 S2&2  |12 q
#k8b1            |1 S1&1  |13 q
#k8b2            |1 S2&2  |13 q
#k9b1            |1 S1&1  |14 q
#k9b2            |1 S2&2  |14 q
#k10b1           |1 S1&1  |15 q
#k10b2           |1 S2&2  |15 q
#k11b1           |1 S1&1  |16 q
#k11b2           |1 S2&2  |16 q
#k12b1           |1 S1&1  |17 q
#k12b2           |1 S2&2  |17 q
#k13b1           |1 S1&1  |18 q
#k13b2           |1 S2&2  |18 q
#k14b1           |1 S1&1  |19 q
#k14b2           |1 S2&2  |19 q
#k15b1           |1 S1&1  |20 q
#k15b2           |1 S2&2  |20 q
#k16b1           |1 S1&1  |21 q
#k16b2           |1 S2&2  |21 q
#k17b1           |1 S1&1  |22 q
#k17b2           |1 S2&2  |22 q
#k18b1           |1 S1&1  |23 q
#k18b2           |1 S2&2  |23 q
#k19b1           |1 S1&1  |24 q
#k19b2           |1 S2&2  |24 q
#k20b1           |1 S1&1  |25 q
#k20b2           |1 S2&2  |25 q
#end-hamiltonian-section
