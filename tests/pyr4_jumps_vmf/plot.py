from pylab import *

fs2au = 41.3413745758

# plot unitary propagation
#for i in range(10):
#    if i==0:
#        da = loadtxt('pyr4_ntraj_1.txt_%d'%(i))
#    else:
#        da += loadtxt('pyr4_ntraj_1.txt_%d'%(i))
#da /= 10.
da = loadtxt('pyr4_ntraj_1.txt_0')
plot(da[:,0]/fs2au,da[:,1],'-r')
plot(da[:,0]/fs2au,da[:,2],'-b')
da = loadtxt('pyr4_jumps_combined_sparse.txt_0')
plot(da[:,0]/fs2au,da[:,1],'--r')
plot(da[:,0]/fs2au,da[:,2],'--b')
da = loadtxt('chk.pl')
plot(da[:,0],da[:,1],'-g')
plot(da[:,0],da[:,2],'-y')
ylim(0.,1.)
xlim(0.,120.)
xlabel('t / fs')
ylabel('pops (t)')
show()
