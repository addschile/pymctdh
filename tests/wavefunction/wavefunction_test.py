import numpy as np
import pymctdh

if __name__ == "__main__":

    ### 4-mode pyrazine model ###
    print('4-mode pyrazine model')
    nel    = 2
    nmodes = 4
    nspfs = np.array([[7, 12, 6, 5],
                     [7, 12, 6, 5]], dtype=int)
    npbfs = np.array([22, 32, 21, 12], dtype=int)
    wf = pymctdh.Wavefunction(nel, nmodes, nspfs, npbfs)
    print(wf.combined)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')

    ### 4-mode pyrazine model list input ###
    print('4-mode pyrazine model list input')
    nel    = 2
    nmodes = 4
    nspfs = [[7, 12, 6, 5],[7, 12, 6, 5]]
    npbfs = [22, 32, 21, 12]
    wf = pymctdh.Wavefunction(nel, nmodes, nspfs, npbfs)
    print(wf.combined)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')

    ### 4-mode pyrazine model with combined modes ###
    print('4-mode pyrazine model with 2 combined modes')
    nel    = 2
    nmodes = 2
    nspfs = np.array([[8, 8],[7, 7]], dtype=int)
    npbfs = [[17, 27],[17, 10]]
    wf = pymctdh.Wavefunction(nel, nmodes, nspfs, npbfs)
    ind0 = wf.psistart[1,0]
    indf = wf.psiend[1,0]
    print(wf.psi[ind0:indf].shape)
    ind0 = wf.psistart[1,1]
    indf = wf.psiend[1,1]
    print(wf.psi[ind0:indf].shape)
    print(wf.combined)
    print(wf.cmodes)
    print(wf.nspfs,type(wf.nspfs))
    print(wf.npbfs,type(wf.npbfs))
    print(wf.spfstart)
    print(wf.spfend)
    wf.generate_ic(1)
    ind = wf.psistart[0,1]
    print(wf.psi[ind])
    print('')
    print('')
