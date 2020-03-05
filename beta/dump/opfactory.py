import numpy as np
import scipy.sparse as sp
from pymctdh.cy.sparsemat import make_csr

#class CSRmat(object):
#    def __init__(self, data, IA, JA):
#        self.data     = data
#        self.IA       = IA
#        self.JA       = JA
#        self.nnz      = self.IA[-1]
#        self.nrows    = len(self.IA)-1
#
#def make_csr(n,op):
#    """Turns a nxn dense matrix to csr format.
#    """
#    data = []
#    IA = []
#    JA = []
#    IA.append( 0 )
#    nnz = 0
#    for i in range(n):
#        JA_tmp = np.nonzero(op[i,:])[0]
#        nnz += len(JA_tmp)
#        IA.append( nnz )
#        data += list(op[i,JA_tmp])
#        JA += list(JA_tmp)
#    _data = np.array(data)
#    _IA = np.array(IA, dtype=np.intc)
#    _JA = np.array(JA, dtype=np.intc)
#    return CSRmat(_data,_IA,_JA)

# NOTE: this currently doesn't get used anywehre
def opdag(op):
    """Function that returns the string representing the conjugate transpose of
    the operator.
    """
    if '*' in op:
        _op = op.split('*')
        opout = opdag(_op[-1])
        for i in range(len(_op)-1):
            opout += '*'+opdag(_op[i-2])
    elif '^' in op:
        _op = op.split('^')
        opout = opdag(_op[0])+'^'+_op[-1]
    elif op == 'adag':
        opout = 'a'
    elif op == 'a':
        opout = 'adag'
    elif op == 'q':
        opout = 'q'
    elif op == 'p':
        opout = 'p'
    elif op == 'KE':
        opout = 'KE'
    elif op == 'n':
        opout = 'n'
    elif op == 'n+1':
        opout = 'n+1'
    else:
        print(op)
        raise ValueError("Not a valid operator for HO basis")
    return opout

def make_ho_ops(params,op,sparse=False):
    """Function that returns the matrix representation of various operators in
    the harmonic oscillator basis.
    """
    npbf   = params['npbf']
    mass   = params['mass']
    omega = params['omega']
    if '*' in op:
        _op = op.split('*')
        opout = make_ho_ops(params,_op[0])
        for i in range(len(_op)-1):
            opout = np.dot(opout,make_ho_ops(params,_op[i+1]))
    elif '^' in op:
        _op = op.split('^')
        opout = make_ho_ops(params,_op[0])
        for i in range(int(_op[-1])-1):
            opout = np.dot(opout,make_ho_ops(params,_op[0]))
    elif op == '1':
        opout = np.eye(npbf)
    elif op == 'adag':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf-1):
            opout[i+1,i] = np.sqrt(float(i+1))
    elif op == 'a':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf-1):
            opout[i,i+1] = np.sqrt(float(i+1))
    elif op == 'q':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf-1):
            opout[i,i+1] = np.sqrt(0.5*float(i+1))
            opout[i+1,i] = np.sqrt(0.5*float(i+1))
    elif op == 'p':
        opout = np.zeros((npbf,)*2, dtype=complex)
        for i in range(npbf-1):
            opout[i,i+1] = 1.j*np.sqrt(0.5*float(i+1))
            opout[i+1,i] = -1.j*np.sqrt(0.5*float(i+1))
    elif op == 'KE':
        opout = make_ho_ops(params,'p')
        opout = 0.5*np.dot(opout,opout)
        opout = opout.real
    elif op == 'n':
        opout = make_ho_ops(params,'adag')
        opout = np.dot(opout,make_ho_ops(params,'a'))
    elif op == 'n+1':
        opout = make_ho_ops(params,'n')
        ndim = opout.shape[0]
        opout += np.eye(ndim)
    else:
        print(op)
        raise ValueError("Not a valid operator for HO basis")
    if sparse:
        #opout = sp.csr_matrix(opout)
        opout = make_csr(int(npbf),opout)
    return opout

def make_ho_ops_combined(params,op,sparse=False):
    """
    """
    npbf   = params['npbf']
    mass   = params['mass']
    omegas = params['omega']
    if '(' and ')' in op:
        ops = op.split(')*(')
        for i in range(len(ops)):
            ops[i] = ops[i].split('(')[-1]
            ops[i] = ops[i].split(')')[0]
            ops[i] = ops[i].split('*')
        for i in range(len(ops)):
            par = {'npbf':npbf[i], 'mass':mass[i], 'omega':omegas[i]}
            for j in range(len(ops[i])):
                if j==0:
                    opout_ = make_ho_ops(par,ops[i][j])
                else:
                    opout_ = np.dot(opout_,make_ho_ops(par,ops[i][j]))
            if i==0:
                opout = opout_
            else:
                opout = np.kron(opout, opout_)
    elif op == '1':
        npbfs   = 1
        for i in range(len(npbf)):
            npbfs *= npbf[i]
        par = {'npbf':npbfs, 'mass':mass[0], 'omega':omegas[0]}
        opout = make_ho_ops(par,op)
    else:
        raise ValueError("Incorrectly specified operator for combined mode")
    if sparse:
        #opout = sp.csr_matrix(opout)
        npbf_ = 1
        for n in npbf:
            npbf_ *= n
        opout = make_csr(npbf_,opout)
    return opout

def make_sinc_ops(params,op,sparse=True):
    """Function that returns the matrix representation of various operators in
    the sinc function basis.

    References
    ----------
    """
    npbf = params['npbf']
    qmin = params['qmin']
    qmax = params['qmax']
    dq   = params['dq']  
    mass = params['mass']
    if op == '1':
        opout = np.eye(npbf)
    elif op == 'q':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf-1):
            opout[i,i+1] = np.sqrt(0.5*float(i+1))
            opout[i+1,i] = np.sqrt(0.5*float(i+1))
    elif op == 'KE':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf):
            for j in range(npbf):
                pre = ((-1.)**float(i-j))/2./mass/dq**2.
                if i==j:
                    opout[i,j] = pre*np.pi**2./3.
                else:
                    opout[i,j] = pre*2./float(i-j)**2.
    else:
        raise ValueError("Not a valid operator for HO basis")
    return opout

def make_planewave_ops(params,op,sparse=True):
    """Function that returns the matrix representation of various operators in
    the plane wave basis.
    """
    npbf = params['npbf']
    nm   = params['nm']
    mass = params['mass']
    if '*' in op:
        _op = op.split('*')
        opout = make_planewave_ops(params,_op[0])
        for i in range(len(_op)-1):
            opout = np.dot(opout,make_planewave_ops(params,_op[i+1]))
    elif '^' in op:
        _op = op.split('^')
        opout = make_ho_ops(params,_op[0])
        for i in range(int(_op[-1])-1):
            opout = np.dot(opout,make_planewave_ops(params,_op[0]))
    elif op == '1':
        opout = np.eye(npbf)
    elif op == 'KE':
        opout = np.zeros((npbf,)*2)
        for i in range(-nm,nm+1):
            opout[i+nm,i+nm] = -float(i)**2.
    elif op == 'cos':
        opout = np.zeros((npbf,)*2)
        for i in range(npbf-1):
            opout[i,i+1] = 0.5
            opout[i+1,i] = 0.5
    if sparse:
        #opout = sp.csr_matrix(opout)
        opout = make_csr(int(npbf),opout)
    return opout

def make_morse_ops(params,op,sparse=True):
    """Function that returns the matrix representation of various operators in
    the morse oscillator basis.

    References
    ----------
    """
    raise NotImplementedError
