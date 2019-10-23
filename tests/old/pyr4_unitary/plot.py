from pylab import *

fs2au = 41.3413745758

# plot unitary propagation
da = loadtxt('pyr4_profile.txt')
plot(da[:,0]/fs2au,da[:,1],'-r')
plot(da[:,0]/fs2au,da[:,2],'-b')
da = loadtxt('chk.pl')
plot(da[:,0],da[:,1],'--g')
plot(da[:,0],da[:,2],'--y')
da = loadtxt('pyr4_lanczos.txt')
plot(da[:,0],da[:,1],'ok',markevery=10)
plot(da[:,0],da[:,2],'oc',markevery=10)
da = loadtxt('pyr4_qutip.txt')
plot(da[:,0],da[:,1],'--k')
plot(da[:,0],da[:,2],'--c')
ylim(0.,1.)
xlim(0.,120.)
xlabel('t / fs')
ylabel('pops (t)')
savefig('pyr4_unitary.png')
show()
