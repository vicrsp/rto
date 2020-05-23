import numpy as np
from src.model.williams_otto import Reactor 
from src.model.semi_batch import SemiBatchReactor 
from gekko import GEKKO
import json
import matplotlib.pyplot as plt
plt.close('all')

r = SemiBatchReactor()
m = r.getModel()

# general options
#results = r.solveMPC(m)
results = r.solveIVP()

plt.plot(results.t, results.y[0])
plt.plot(results.t, results.y[1])
plt.plot(results.t, results.y[2])
plt.plot(results.t, results.y[3])
plt.legend(['Ca','Cb','Cc','Cd'])
plt.show()

plt.figure()
plt.plot(results.t, results.y[4])
plt.legend('V')
plt.show()
#r.solveRTO(m)

# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True)
# ax1.plot(results['time'], results['ca'])
# ax1.legend(['Ca'])

# ax2.plot(results['time'], results['cb'])
# ax2.legend(['Cb'])

# ax3.plot(results['time'], results['cc'])
# ax3.legend(['Cc'])

# ax4.plot(results['time'], results['cd'])
# ax4.legend(['Cd'])

# ax5.plot(results['time'], results['f'])
# ax5.legend(['F'])
# plt.show()



