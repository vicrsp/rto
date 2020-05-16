import numpy as np
from src.model.williams_otto import Reactor 
from gekko import GEKKO
import json
import matplotlib.pyplot as plt
plt.close('all')

r = Reactor()
m = r.getModel()

# general options
m.time = np.linspace(0,25,100)
m.options.CV_TYPE = 2 # squared error
m.options.IMODE = 6 # control

m.solve(disp=False)
with open(m.path+'//results.json') as f:
    results = json.load(f)

#print(results.keys())

# plt.figure()
# plt.plot(results['time'], results['k1'])
# plt.plot(results['time'], results['k2'])
# plt.plot(results['time'], results['k3'])
# plt.legend(['k1','k2','k3'])

# #plt.plot()
# #plt.plot(results['time'], results['m'])
# #plt.plot(results['time'], results['v'])

plt.figure()
plt.plot(results['time'], results['tr'])
plt.legend(['tr'])

plt.figure()
plt.plot(results['time'], results['fa'])
plt.plot(results['time'], results['fb'])
plt.legend(['fa','fb'])
plt.show()

plt.figure()
plt.plot(results['time'], results['ma'])
plt.plot(results['time'], results['mb'])
plt.plot(results['time'], results['mc'])
plt.plot(results['time'], results['me'])
plt.plot(results['time'], results['mp'])
plt.plot(results['time'], results['mg'])
#plt.plot(results['time'], results['mu'])
plt.legend(['ma','mb','mc','me','mp','mg'])
plt.show()

plt.figure()
plt.plot(results['time'], results['ma'])
plt.plot(results['time'], results['ma.tr'])
plt.plot(results['time'], results['ma.sp'])
plt.legend(['ma','tr','sp'])
plt.show()

plt.figure()
plt.plot(results['time'], results['mg'])
plt.plot(results['time'], results['mg.tr'])
plt.plot(results['time'], results['mg.sp'])
plt.legend(['mg','tr','sp'])
plt.show()





