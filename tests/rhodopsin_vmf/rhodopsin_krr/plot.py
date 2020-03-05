from pylab import *

fs2au = 41.341374575

da = loadtxt('rhodopsin_diabatic_pops.txt')
plot(da[:,0]/fs2au/1000.,da[:,1],'-r')
plot(da[:,0]/fs2au/1000.,da[:,2],'-b')
da = loadtxt('../thoss_model_ci_no_bath.dat')
plot(da[:,0]/fs2au/1000.,da[:,1],'--c')
plot(da[:,0]/fs2au/1000.,da[:,2],'--y')
ylim(0,1)
xlim(0,4.)
xlabel(r'$t$ / ps',size=15)
ylabel(r'$P(t)$',size=15)
savefig('rhodopsin_db_pops_krr.png')
show()
