from qutip import *
from pylab import *

H = sigmaz()
L = destroy(2)
psi0 = basis(2,0)
times = arange(0.0,10.,0.1)
results = mcsolve(H, psi0, times, c_ops=[L], e_ops=[sigmaz()], ntraj=1)
print(results.col_times)
print(results.col_which)
plot(times,results.expect[0])
show()
