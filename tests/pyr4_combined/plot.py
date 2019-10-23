from pylab import *

fs2au = 41.3413745758

# plot unitary propagation
da = loadtxt('pyr4_profile_sparse.txt')
plot(da[:,0]/fs2au,da[:,1],'-r')
plot(da[:,0]/fs2au,da[:,2],'-b')
da = loadtxt('chk.pl')
plot(da[:,0],da[:,1],'--g')
plot(da[:,0],da[:,2],'--y')
ylim(0.,1.)
xlim(0.,120.)
xlabel('t / fs')
ylabel('pops (t)')
show()
