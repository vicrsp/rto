import numpy as np
from src.model.williams_otto import Reactor 
from gekko import GEKKO
import json
import matplotlib.pyplot as plt
plt.close('all')

r = Reactor()
m = r.getModel()

# general options
results = r.solveMPC(m)
r.solveRTO(m)

plt.figure()
plt.plot(results['time'], results['tr'])
plt.legend(['tr'])

plt.figure()
plt.plot(results['time'], results['fa'])
plt.plot(results['time'], results['fb'])
plt.legend(['fa','fb'])
plt.show()

plt.figure()
plt.plot(results['time'], results['xa'])
plt.plot(results['time'], results['xb'])
plt.plot(results['time'], results['xc'])
plt.plot(results['time'], results['xe'])
plt.plot(results['time'], results['xp'])
plt.plot(results['time'], results['xg'])
#plt.plot(results['time'], results['mu'])
plt.legend(['xa','xb','xc','xe','xp','xg'])
plt.show()

plt.figure()
plt.plot(results['time'], results['xa'])
plt.plot(results['time'], results['xa.tr'])
plt.plot(results['time'], results['xa.sp'])
plt.legend(['xa','tr','sp'])
plt.show()

plt.figure()
plt.plot(results['time'], results['xg'])
plt.plot(results['time'], results['xg.tr'])
plt.plot(results['time'], results['xg.sp'])
plt.legend(['xg','tr','sp'])
plt.show()





