import numpy as np
from copy import deepcopy
from .qoperator import QOperator
from .expect import compute_expect,diabatic_pops,diabatic_grid_pops
from .log import print_basic

class Results(object):
    """
    Results class that helps organize and print out relevant results.
    """

    def __init__(self, nsteps=None, db_pops=False, print_db_pops=False, 
                 db_pops_file=None, db_grid_pops=False, print_db_grid_pops=False,
                 db_grid_pops_file=None, db_grid_pops_modes=None, e_ops=None, 
                 print_es=False, es_file=None, store_states=False, store_every=1,
                 print_restart=False, restart_file=None, restart_every=1,
                 print_states=False, states_file=None, states_every=1,
                 jump_stats=False):
        """Initialize results class.

        Attributes
        ----------
        nsteps - int, total number of simulation steps
        e_ops - list of strings or pymctdh.qoperators
        print_es - bool, flag to print or not print expectation values
        es_file - string, filename for printing of expectation values
        norm - float, normalization coefficient used for averaging trajectories
        store_states - bool, flag to store each state in a list
        print_restart - bool, flag to print state into a restart
        restart_file - string, filename for the restart state
        restart_every - int, determines how often restart state is printed
        print_states - bool, flag to print each state
        states_file - string, filename for printing the states
        states_every - int, determines how often states are printed

        Methods
        -------
        analyze_state
        close_down
        """

        self.nsteps = nsteps

        # diabatic populations containers #
        self.db_pops = db_pops
        self.db = None
        self.print_db_pops = print_db_pops
        self.fdb = None
        self.db_pops_file = db_pops_file

        # diabatic grid populations containers #
        self.db_gpops = db_grid_pops
        self.print_db_gpops = print_db_grid_pops
        self.db_gpops_file = db_grid_pops_file
        self.db_gpops_modes = db_grid_pops_modes

        # TODO
        # expectation value containers #
        self.expect = None
        self.print_es = print_es
        self.fes = None
        self.es_file = es_file
        self.norm = 0.0
        #if e_ops != None:
        #    if isinstance(e_ops, list):
        #        self.e_ops = e_ops
        #    else:
        #        self.e_ops = [e_ops]
        #    self.expect = np.zeros((self.nsteps,len(self.e_ops)))
        #else:
        #    self.e_ops = e_ops

        # states containers #
        self.store_states = store_states
        self.states = None
        if self.store_states:
            self.states = list()

        # print restart state info #
        self.print_restart = print_restart
        self.restart_file = None
        if self.print_restart:
            self.restart_file = restart_file
        self.restart_every = restart_every

        # print states info #
        self.print_states = print_states
        self.states_file = None
        if self.print_states:
            self.states_file = states_file
        self.states_every = states_every

        # TODO
        ## jump statistics containers #
        #self.jump_stats = jump_stats
        #self.jumps = None
        #self.jump_times = None
        #if self.jump_stats:
        #    self.jumps = list()

    def add(self, res, weight=1.0):
        """Function that adds results from a results class to another results class.
        Used for any dynamics that has to sample over trajectory realizations.

        Parameters
        ----------
        res - Results class, results to add to current results
        weight - float, number by which to weight the results being added
        """
        if self.store_states:
            self.states += res.states
        if not self.db is None:
            self.db += weight*res.db
            self.norm += weight
        #if not self.e_ops is None:
        #    self.expect += weight*res.expect
        #    self.norm += weight
        #if not self.jumps is None:
        #    self.jumps += res.jumps

    def average(self):
        if not self.db is None:
            self.db /= self.norm
        #if not self.e_ops is None:
        #    self.expect /= self.norm

    def copy(self, traj=None):
        res_out = deepcopy(self)
        if not traj is None:
            # update files
            if not self.db_pops_file is None:
                res_out.db_pops_file += "_%d"%(traj)
            if not self.es_file is None:
                res_out.es_file += "_%d"%(traj)
            if not self.states_file is None:
                res_out.states_file += "_%d"%(traj)
        return res_out


    def close_down(self):
        if self.fes != None and self.fes.closed == False:
            self.fes.close()

    #def compute_expectation(self, ind, state, si):
    #    """Computes expectation values from qoperator.
    #    """
    #    for i,e_op in enumerate(self.e_ops):
    #        self.expect[ind,i] = compute_expect(si.nel,si.nmodes,si.nspfs,
    #                                si.npbfs,si.spfstart,si.spfend,
    #                                si.psistart,si.psiend,state,e_op,pbfs) # TODO need to get pbfs passed here
    #        if self.print_es:
    #            self.fes.write('%.8f '%(self.expect[i,ind]))

    def print_restart_state(self, state):
        np.save(self.restart_file, state)

    def print_state(self, ind, time, state):
        np.save(self.states_file+"_"+str(ind), state)

    def analyze_state(self, ind, time, state, si, pbfs):
        """Functional interface between propagation and results class.
        ind - int
        time - float
        state - 
        si - list
        """
        
        if self.store_states:
            self.states.append( state.copy() )

        if self.print_restart:
            if ind%self.restart_every==0:
                self.print_restart_state(state)
                print_basic("last state printed: %d, %.8f"%(ind,time))

        if self.print_states:
            if ind%self.states_every==0:
                self.print_state(ind, time, state)

        if self.db_pops:
            if self.print_db_pops: 
                if self.fdb==None:
                    # no file is open need to open one
                    if self.db_pops_file == None:
                        self.fdb = open("diabatic_pops.dat","w")
                    else:
                        self.fdb = open(self.db_pops_file,"w")
                self.fdb.write('%.8f '%(time))
            if self.db is None:
                self.db = np.zeros((self.nsteps,si['nel']))
            self.db[ind,:] = diabatic_pops(si['nel'],si['psistart'],si['psiend'],state)
            if self.print_db_pops:
                for i in range(si['nel']):
                    self.fdb.write('%.8f '%(self.db[ind,i]))
                self.fdb.write('\n')
                self.fdb.flush()

        if self.db_gpops:
            if self.print_db_gpops: 
                if self.db_gpops_file == None:
                   self.db_gpops_file = 'diabatic_grid_pops'
            dbgpops = diabatic_grid_pops(si['nel'],si['nmodes'],si['nspfs'],
                        si['npbfs'],si['psistart'],si['psiend'],si['spfstart'],
                        si['spfend'],pbfs,state,modes=self.db_gpops_modes)
            if self.db_gpops_modes is None:
                for mode in range(si['nmodes']):
                    np.save(self.db_gpops_file+'_mode_%d_%d.npy'%(mode,ind),dbgpops[mode],allow_pickle=True)
            else:
                for i,mode in enumerate(self.db_gpops_modes):
                    np.save(self.db_gpops_file+'_mode_%d_%d.npy'%(mode,ind),dbgpops[i],allow_pickle=True)

        #if self.e_ops != None:
        #    if self.print_es: 
        #        if self.fes==None:
        #            # no file is open need to open one
        #            if self.es_file==None:
        #                self.fes = open("output.dat","w")
        #            else:
        #                self.fes = open(self.es_file,"w")
        #        self.fes.write('%.8f '%(time))
        #    self.compute_expectation(ind, state, state_info)
        #    if self.print_es: 
        #        self.fes.write('\n')
        #        self.fes.flush()

    def store_jumps(self, njumps, jumps):
        self.jumps.append( [njumps, jumps.copy()] )

    def print_expectation(self, es_file=None):
        if es_file==None:
            np.savetxt('expectation_values', self.expect.T)
        else:
            np.savetxt(es_file, self.expect.T)
