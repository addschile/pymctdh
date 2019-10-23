from pylab import *

fs2au=41.3413745758
#ntraj = 102
#for traj in range(ntraj):
#    #da = np.loadtxt('pyr4_jumps_traj_%d.txt'%(traj))
#    #plot(da[:,0]/fs2au,da[:,1],'-r',alpha=0.1)
#    #plot(da[:,0]/fs2au,da[:,2],'-b',alpha=0.1)
#    if traj == 0:
#        da_avg = np.loadtxt('pyr4_jumps_traj_%d.txt'%(traj))
#    else:
#        da_avg += np.loadtxt('pyr4_jumps_traj_%d.txt'%(traj))
#da_avg /= float(ntraj)
#plot(da_avg[:,0]/fs2au,da_avg[:,1],'-r',lw=3)
#plot(da_avg[:,0]/fs2au,da_avg[:,2],'-b',lw=3)
#da = loadtxt('stronger/pyr4_jumps_traj_0.txt')
da = loadtxt('stronger/pyr4_jumps_down_traj_0.txt')
#da = loadtxt('stronger/pyr4_jumps_up_traj_0_old.txt')
#plot(da[:,0],da[:,1],'--r')
#plot(da[:,0],da[:,2],'--b')
#da = loadtxt('stronger/pyr4_jumps_up_traj_0.txt')
plot(da[:,0],da[:,1],'-r')
plot(da[:,0],da[:,2],'-b')
#da = loadtxt('qutip_run/pyr4_qutip.txt')
da = loadtxt('qutip_run/pyr4_qutip_down.txt')
#da = loadtxt('qutip_run/pyr4_qutip_up.txt')
plot(da[:,0]*fs2au,da[:,1],'-g')
plot(da[:,0]*fs2au,da[:,2],'-y')
#da = loadtxt('chk_dm.pl')
#plot(da[:,0],da[:,1],'--or',markevery=20)
#plot(da[:,0],da[:,2],'--ob',markevery=20)
#da = loadtxt('chk.pl')
#plot(da[:,0],da[:,1],'-g')
#plot(da[:,0],da[:,2],'-y')
show()
