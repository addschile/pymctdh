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
                pbfs[i].make_operators(self.ops[i])

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
        self.huterms   = []
        self.huelterms = []
        self.hcterms   = []
        self.hcelterms = []
        for hterm in self.hterms:
            # uncorrelated terms must be electronically diagonal
            if isdiag(hterm['elop']):
                if 'modes' in hterm:
                    if len(hterm['modes'])==1:
                        self.huterms.append( hterm )
                    else:
                        self.hcterms.append( hterm )
                else:
                    self.huelterms.append( hterm )
            else:
                if 'modes' in hterm:
                    self.hcterms.append( hterm )
                else:
                    self.hcelterms.append( hterm )

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

if __name__ == "__main__":

    ### 4-mode pyrazine model ###
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
    #hterms.append({'coeff':    lamda, 'units': 'ev', 'elop':  'sx'}) # Peierls copuling
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': 'q'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 2 el 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 2, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 2, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 3, 'elop': '0,0', 'ops': 'q'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 3, 'elop': '1,1', 'ops': 'q'}) # Holstein copuling mode 4 el 1

    ham = Hamiltonian(nmodes, hterms)
    print('4-mode pyrazine')
    print('ops')
    print(ham.ops)
    print('')
    print('correlated terms')
    print(ham.hcterms)
    print('')
    print('correlated electronic terms')
    print(ham.hcelterms)
    print('')
    print('uncorrelated terms')
    print(ham.huterms)
    print('')
    print('uncorrelated electronic terms')
    print(ham.huelterms)
    print('')
    print('')

    ### 4-mode pyrazine model with combined modes ###
    nmodes =  2
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
    # combined mode 0
    hterms.append({'coeff': 1.0*w10a, 'units': 'ev', 'modes': 0, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff': 0.5*w10a, 'units': 'ev', 'modes': 0, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w6a, 'units': 'ev', 'modes': 0, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w6a, 'units': 'ev', 'modes': 0, 'ops': '(1)*(q^2)'})
    # combined mode 1
    hterms.append({'coeff':   1.0*w1, 'units': 'ev', 'modes': 1, 'ops':  '(KE)*(1)'}) # mode 00 terms
    hterms.append({'coeff':   0.5*w1, 'units': 'ev', 'modes': 1, 'ops': '(q^2)*(1)'})
    hterms.append({'coeff':  1.0*w9a, 'units': 'ev', 'modes': 1, 'ops':  '(1)*(KE)'}) # mode 01 terms
    hterms.append({'coeff':  0.5*w9a, 'units': 'ev', 'modes': 1, 'ops': '(1)*(q^2)'})
    #hterms.append({'coeff':    lamda, 'units': 'ev', 'elop':  'sx'}) # Peierls copuling
    # combined mode 0
    hterms.append({'coeff':    lamda, 'units': 'ev', 'modes': 0, 'elop':  'sx', 'ops': '(q)*(1)'}) # Peierls copuling
    hterms.append({'coeff':     k6a1, 'units': 'ev', 'modes': 0, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 0
    hterms.append({'coeff':     k6a2, 'units': 'ev', 'modes': 0, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 2 el 1
    # combined mode 1
    hterms.append({'coeff':      k11, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 0
    hterms.append({'coeff':      k12, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(q)*(1)'}) # Holstein copuling mode 3 el 1
    hterms.append({'coeff':     k9a1, 'units': 'ev', 'modes': 1, 'elop': '0,0', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 0
    hterms.append({'coeff':     k9a2, 'units': 'ev', 'modes': 1, 'elop': '1,1', 'ops': '(1)*(q)'}) # Holstein copuling mode 4 el 1

    print('4-mode pyrazine combined modes')
    ham = Hamiltonian(nmodes, hterms)
    print('ops')
    print(ham.ops)
    print('')
    print('correlated terms')
    print(ham.hcterms)
    print('')
    print('correlated electronic terms')
    print(ham.hcelterms)
    print('')
    print('uncorrelated terms')
    print(ham.huterms)
    print('')

