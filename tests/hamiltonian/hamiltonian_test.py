import numpy as np
import pymctdh

if __name__ == "__main__":

    ### 4-mode pyrazine model ###
    nel    =  2
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

    ham = pymctdh.Hamiltonian(nel, nmodes, hterms)
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
    for mode in range(nmodes):
        for alpha in range(nel):
            print(ham.huterms[mode][alpha])
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
    ham = pymctdh.Hamiltonian(nel, nmodes, hterms)
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
    for mode in range(nmodes):
        for alpha in range(nel):
            print(ham.huterms[mode][alpha])
    print('')
    print('uncorrelated electronic terms')
    print(ham.huelterms)
    print('')
    print('')

