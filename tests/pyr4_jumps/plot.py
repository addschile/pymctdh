from pylab import *

fs2au = 41.3413745758

# plot unitary propagation
da = loadtxt('pyr4_jumps_down_single_traj.txt')
plot(da[:,0]/fs2au,da[:,1],'-r')
plot(da[:,0]/fs2au,da[:,2],'-b')
da = loadtxt('pyr4_jumps_down_single_traj_old.txt')
plot(da[:,0]/fs2au,da[:,1],'--g')
plot(da[:,0]/fs2au,da[:,2],'--y')
ylim(0.,1.)
xlim(0.,120.)
xlabel('t / fs')
ylabel('pops (t)')
show()
