from pylab import *

fs2au=41.3413745758
ntraj = 43
for traj in range(ntraj):
    if traj == 0:
        da_avg = np.loadtxt('stronger/pyr4_jumps_down_traj_%d.txt'%(traj))
    else:
        da_avg += np.loadtxt('stronger/pyr4_jumps_down_traj_%d.txt'%(traj))
da_avg /= float(ntraj)
plot(da_avg[:,0]/fs2au,da_avg[:,1],'-r',lw=3,label='My MCTDH')
plot(da_avg[:,0]/fs2au,da_avg[:,2],'-b',lw=3)
da = loadtxt('qutip_run/pyr4_qutip_down.txt')
plot(da[:,0],da[:,1],'--r',label='qutip jumps')
plot(da[:,0],da[:,2],'--b')
da = loadtxt('chk.pl')
plot(da[:,0],da[:,1],'-g',label='MCTDH unitary')
plot(da[:,0],da[:,2],'-y')
xlim(0,120)
ylim(0,1)
xlabel(r'$t$ / fs',size=15)
ylabel(r'$P (t)$ (diabatic)',size=15)
legend(loc='best')
savefig('prop_jumps_down.png')
show()

for traj in range(ntraj):
    da = np.loadtxt('stronger/pyr4_jumps_down_traj_%d.txt'%(traj))
    plot(da[:,0]/fs2au,da[:,1],'-r',alpha=0.1)
    plot(da[:,0]/fs2au,da[:,2],'-b',alpha=0.1)
    if traj == 0:
        da_avg = np.loadtxt('stronger/pyr4_jumps_down_traj_%d.txt'%(traj))
    else:
        da_avg += np.loadtxt('stronger/pyr4_jumps_down_traj_%d.txt'%(traj))
da_avg /= float(ntraj)
plot(da_avg[:,0]/fs2au,da_avg[:,1],'-r',lw=3)
plot(da_avg[:,0]/fs2au,da_avg[:,2],'-b',lw=3)
xlim(0,120)
ylim(0,1)
xlabel(r'$t$ / fs',size=15)
ylabel(r'$P (t)$ (diabatic)',size=15)
savefig('prop_jumps_trajs.png')
show()
