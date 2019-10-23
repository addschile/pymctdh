#from __future__ import print_function,absolute_import
import numpy as np
#from .log import print_basic
from log import print_basic

def add_results(results1, results2, weight=None):
    """
    Function that adds results from a results class to another results class.
    Used for any dynamics that has to sample over trajectory realizations.

    Parameters
    ----------
    results1: Results class
        Container results for averaging.
    results2: Results class
        Results to add to the container.
    weight: TODO
    """
    if results1==None:
        from copy import deepcopy
        return deepcopy(results2)
    else:
        if results1.e_ops != None:
            results1.expect += results2.expect
        if results1.map_ops:
            results1.maps += results2.maps
        if results1.jumps != None:
            results1.jumps += results2.jumps
        return results1

def avg_results(ntraj, results):
    if results.e_ops != None:
        results.expect /= float(ntraj)
    if results.map_ops:
        results.maps /= float(ntraj)
    return results

class Results(object):
    """
    Results class that helps organize and print out relevant results.
    """

    def __init__(self, tobs=None, e_ops=None, print_es=False, es_file=None, 
                 print_final=False, final_file=None, final_every=1,
                 print_states=False, states_file=None, states_every=1,
                 jump_stats=False, every=1):
        """
        Initialize results class.

        Parameters
        ----------
        e_ops: list of np.ndarrays
        store_states: bool
        print_states: bool
        states_file: string
        jump_stats: bool
        every: int
        """
        self.tobs = tobs
        # how often do we compute results #
        self.every = every
        # expectation value containers #
        self.expect = None
        self.print_es = print_es
        self.fes = None
        self.es_file = es_file
        if e_ops != None:
            if isinstance(e_ops, list):
                self.e_ops = e_ops
            else:
                self.e_ops = [e_ops]
            self.expect = np.zeros((len(self.e_ops),tobs))
        else:
            self.e_ops = e_ops
        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()
        # print dynamic state info #
        self.print_final = print_final
        self.final_file = None
        if self.print_final:
            self.final_file = final_file
        self.final_every = final_every
        # print states info #
        self.print_states = print_states
        self.states_file = None
        if self.print_states:
            self.states_file = states_file
        self.states_every = states_every
        # jump statistics containers #
        self.jump_stats = jump_stats
        self.jumps = None
        self.jump_times = None
        if self.jump_stats:
            self.jumps = list()

    def close_down(self):
        """
        """
        if self.fes != None and self.fes.closed == False:
            self.fes.close()

    # TODO
    def compute_expectation(self, ind, state, normalized=False):
        """Computes expectation values.
        """

    def print_final_state(self, wf):
        """
        """
        np.save(self.final_file, wf.psi)

    def print_state(self, ind, time, wf):
        """
        """
        np.save(self.states_file+"_"+str(ind), wf.psi)

    def analyze_state(self, ind, time, wf):
        """Functional interface between propagation and results class
        """
        if self.store_states:
            self.states.append( wf.psi.copy() )
        if self.print_final:
            if ind%self.final_every==0:
                self.print_final_state(wf)
                print_basic("last state printed: %d, %.8f"%(ind,time))
        if self.print_states:
            if ind%self.states_every==0:
                self.print_state(ind, time, wf)
        # TODO
        #if self.e_ops != None:
        #    if self.print_es: 
        #        if self.fes==None:
        #            # no file is open need to open one
        #            if self.es_file==None:
        #                self.fes = open("output.dat","w")
        #            else:
        #                self.fes = open(self.es_file,"w")
        #        self.fes.write('%.8f '%(time))
        #    self.compute_expectation(ind, state)
        #    if self.print_es: 
        #        self.fes.write('\n')
        #        self.fes.flush()
        if ind==(self.tobs-1):
            self.close_down()

    def store_jumps(self, njumps, jumps):
        """
        """
        self.jumps.append( [njumps, jumps.copy()] )

    def print_expectation(self, es_file=None):
        """
        """
        if es_file==None:
            np.savetxt('expectation_values', self.expect.T)
        else:
            np.savetxt(es_file, self.expect.T)
