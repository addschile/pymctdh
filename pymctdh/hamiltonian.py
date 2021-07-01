import numpy as np
import pymctdh.units as units
from .optools import isdiag

class Hamiltonian(object):

    def __init__(self, nel, nmodes, hterms, pbfs=None):
        """
        """
        # number of electronic states
        self.nel = nel
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
            for i in range(nmodes):
                # make all the operators for the pbf
                pbfs[i].make_operators(self.ops[i])
                if not self.huterms is None:
                    # make the 1-body hamiltonian for the pbf
                    pbfs[i].make_1b_ham(self.nel, self.huterms[i])

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
        huterms        = []
        self.huterms   = []
        self.huelterms = []
        self.hcterms   = []
        self.hcelterms = []
        for hterm in self.hterms:
            # uncorrelated terms must be electronically diagonal
            if isdiag(hterm['elop']):
                if 'modes' in hterm:
                    if len(hterm['modes'])==1:
                        huterms.append( hterm )
                    else:
                        self.hcterms.append( hterm )
                else:
                    self.huelterms.append( hterm )
            else:
                if 'modes' in hterm:
                    self.hcterms.append( hterm )
                else:
                    self.hcelterms.append( hterm )
        if len(huterms) == 0:
            self.huterms = None
        else:
            # make single-body terms specifically
            self.huterms = [[[] for i in range(self.nel)] for i in range(self.nmodes)]
            for hterm in huterms:
                mode = hterm['modes'][0]
                elop = hterm['elop']
                if elop == '1':
                    for alpha in range(self.nel):
                        self.huterms[mode][alpha].append( hterm )
                elif elop == 'sz':
                    self.huterms[mode][0].append( hterm )
                    hterm['coeff'] *= -1.
                    self.huterms[mode][1].append( hterm )
                else:
                    alpha = int(elop[0])
                    self.huterms[mode][alpha].append( hterm )

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
            else:
                for mode in range(self.nmodes):
                    self.ops[mode].append( '1' )
