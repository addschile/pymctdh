from pylab import *

fs2au = 41.3413745758

# plot unitary propagation
da = loadtxt('pyr4_jumps.txt')
plot(da[:,0]/fs2au,da[:,1],'-r')
plot(da[:,0]/fs2au,da[:,2],'-b')
#da = loadtxt('tests/chk.pl')
#plot(da[:,0],da[:,1],'--g')
#plot(da[:,0],da[:,2],'--y')
da = loadtxt('pyr4_arnoldi_jumps.txt_traj_0')
plot(da[:,0]/fs2au,da[:,1],'--r')
plot(da[:,0]/fs2au,da[:,2],'--b')
ylim(0.,1.)
xlim(0.,120.)
xlabel('t / fs')
ylabel('pops (t)')
#savefig('pyr4_unitary.png')
show()
