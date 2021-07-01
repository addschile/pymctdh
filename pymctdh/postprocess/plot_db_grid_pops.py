import numpy as np
import matplotlib.pyplot as plt

def plot_db_grid_pops(nel, nmodes, ns, times, grids, every=1, filename=None, 
                      plottype='contour', cmap='coolwarm'):
    """
    """
    if filename is None:
        filename = 'diabatic_grid_pops'

    # load the data
    qs = np.zeros((len(times),nel,nmodes))
    for i in range(len(times)):
        qt = np.load(filename+'_%d.npy'%(every*i))
        for j in range(nel):
            qs[i,j,:] = qt[i][j,:]

    if plottype == 'contour':
        # plot a contour map for each mode
        for i in range(nmodes):
            TT,QQ = np.meshgrid(grids[i],times)
            for j in range(nel):
                plt.subplot(1,nel,j+1)
                plt.contourf(TT,QQ,qs[:,j,:],cmap=cmap)
            plt.show()
    return
