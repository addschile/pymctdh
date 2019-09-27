import numpy as np

def opdag(op):
    """Function that returns the string representing the conjugate transpose of
    the operator.
    """
    if '^' in op:
        _op = op.split('^')
        opout = opdag(_op[0])+'^'+_op[-1]
    elif '*' in op:
        _op = op.split('*')
        opout = opdag(_op[0])
        for i in range(len(_op)-1):
            opout += '*'+opdag(_op[i+1])
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
    else:
        raise ValueError("Not a valid operator for HO basis")
    return opout

def make_ho_ops(params,op):
    """Function that returns the matrix representation of various operators in
    the harmonic oscillator basis.
    """
    npbf   = params['npbf']
    mass   = params['mass']
    omegas = params['omega']
    if '^' in op:
        _op = op.split('^')
        opout = make_ho_ops(params,_op[0])
        for i in range(int(_op[-1])-1):
            opout = np.dot(opout,make_ho_ops(params,_op[0]))
    elif '*' in op:
        _op = op.split('*')
        opout = make_ho_ops(params,_op[0])
        for i in range(len(_op)-1):
            opout = np.dot(opout,make_ho_ops(params,_op[i+1]))
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
    elif op == 'n':
        opout = make_ho_ops(params,'adag')
        opout = np.dot(opout,make_ho_ops(params,'a'))
    else:
        raise ValueError("Not a valid operator for HO basis")
    return opout

def make_sinc_ops(params,op):
    """
    """
    npbf = params['npbf']
    qmin = params['qmin']
    qmax = params['qmax']
    dq   = params['dq']  
    mass = params['mass']
    if op == 'q':
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

def make_planewave_ops(params,op):
    """
    """
    raise NotImplementedError
