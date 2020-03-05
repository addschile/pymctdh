import numpy as np
import pymctdh

if __name__ == "__main__":

    print('combined mode version')
    nel    = 2
    nmodes = 2
    nspfs = np.array([[10,10],
                      [8,8]], dtype=int)
    npbfs = [[18, 28],[18, 14]]

    pbfs = list()
    mode1 = [['ho', npbfs[0][0], 1.0, 1.0],['ho', npbfs[0][1], 1.0, 1.0]]
    mode2 = [['ho', npbfs[1][0], 1.0, 1.0],['ho', npbfs[1][1], 1.0, 1.0]]
    pbfs.append( pymctdh.PBasis(mode1, combined=True, sparse=True) )
    pbfs.append( pymctdh.PBasis(mode2, combined=True, sparse=True) )

    gam = 1.0
    term = {'coeff': np.sqrt(gam), 'units': 'au', 'modes': 0, 'elop': '1', 'ops': '(a)*(1)'}
    L = pymctdh.QOperator(nmodes, term, pbfs=pbfs)
#    # make Lindblad operators
#    Ls = []
#    term = {'coeff': np.sqrt(gam), 'units': 'au', 'modes': 0, 'elop': '1', 'ops': '(a)*(1)'}
#    Ls.append( pymctdh.QOperator(nmodes, term, pbfs=pbfs) )
#    term = {'coeff': np.sqrt(gam), 'units': 'au', 'modes': 0, 'elop': '1', 'ops': '(1)*(a)'}
#    Ls.append( pymctdh.QOperator(nmodes, term, pbfs=pbfs) )
#    term = {'coeff': np.sqrt(gam), 'units': 'au', 'modes': 1, 'elop': '1', 'ops': '(a)*(1)'}
#    Ls.append( pymctdh.QOperator(nmodes, term, pbfs=pbfs) )
#    term = {'coeff': np.sqrt(gam), 'units': 'au', 'modes': 1, 'elop': '1', 'ops': '(1)*(a)'}
#    Ls.append( pymctdh.QOperator(nmodes, term, pbfs=pbfs) )
#
    #print('making LdL operators')
    ## make Lindblad waiting time operators
    #LdLs = []
    #term = {'coeff': gam, 'units': 'au', 'modes': 0, 'elop': '1', 'ops': '(n)*(1)'}
    #LdLs.append( QOperator(nmodes, term, pbfs=pbfs) )
    #term = {'coeff': gam, 'units': 'au', 'modes': 0, 'elop': '1', 'ops': '(1)*(n)'}
    #LdLs.append( QOperator(nmodes, term, pbfs=pbfs) )
    #term = {'coeff': gam, 'units': 'au', 'modes': 1, 'elop': '1', 'ops': '(n)*(1)'}
    #LdLs.append( QOperator(nmodes, term, pbfs=pbfs) )
    #term = {'coeff': gam, 'units': 'au', 'modes': 1, 'elop': '1', 'ops': '(1)*(n)'}
    #LdLs.append( QOperator(nmodes, term, pbfs=pbfs) )
