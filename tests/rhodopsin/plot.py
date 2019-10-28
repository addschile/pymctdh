from pylab import *

fs2au = 41.341374575

#da = loadtxt('rhodopsin_diabatic_pops_test.txt')
da = loadtxt('rhodopsin_diabatic_pops_really_small.txt')
plot(da[:,0]/fs2au/1000.,da[:,1],'-r')
plot(da[:,0]/fs2au/1000.,da[:,2],'-b')
#da = loadtxt('rhodopsin_diabatic_pops_small.txt')
#plot(da[:,0]/fs2au/1000.,da[:,1],'or',markevery=10)
#plot(da[:,0]/fs2au/1000.,da[:,2],'ob',markevery=10)
da = loadtxt('thoss_model_ci_no_bath.dat')
plot(da[:,0]/fs2au/1000.,da[:,1],'--c')
plot(da[:,0]/fs2au/1000.,da[:,2],'--y')
ylim(0,1)
#xlim(0,0.5)
xlim(0,2.)
show()
