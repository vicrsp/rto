import numpy as np
from gekko import GEKKO
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# CONSTANTS
R = 8.31446261815324 # gas constant
g = 9.80665 # [m/s2]

L_w = 1500 # [m]
H_w = 1000 # [m]
D_w = 0.121 # [m]

L_bh = 500 # [m]
H_bh = 100 # [m]
D_bh = 0.121 # [m]

L_a = 1500 # [m]
H_a = 1000 # [m]
D_a = 0.189 # [m]
A_a = math.pi * (D_a / 2)**2 # [m2]

A_w = math.pi*((D_w/2)**2)
A_bh = math.pi*((D_bh/2)**2)
V_a = L_a * (math.pi*(D_a / 2)**2 - math.pi*(D_w / 2)**2)

rho_o = 900 # [kg/m3]
C_iv = 1e-4 # [m2]
C_pc = 1e-3 # [m2]
p_m = 20 # [bar]
p_res = 150 # [bar]
PI = 2.2 # [kg/s/bar]
T_a = 28 + 273 # [ºK]
T_w = 32 + 273 # [ºK]
Mw = 20e-3 # [g]
GOR = 0.1 # [kg/kg]
mu_oil = 0.001 # 1cP

class GasLiftedWell:
    def __init__(self, name, k=(2.2, 0.1, 900), y0=[1, 1, 1]):
      self.name = name
      self.y0 = y0
      self.PI, self.GOR, self.rho_o = k
      self.stoptime = 300
      self.numpoints = 100
      
    def calculate_flows(self, x):
      m_ga, m_gt, m_ot = x # current state
      
      p_ai = 1e-5*(((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3) + (Mw/(R*T_a)*((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3))*g*H_a)
      p_wh = 1e-5*(((R*T_w/Mw)*(m_gt*1e3/(L_w*A_w + L_bh*A_bh - m_ot*1e3/self.rho_o))) - ((m_gt*1e3+m_ot*1e3 )/(L_w*A_w))*g*H_w/2)
      rho_ai = 1e-2*(Mw/(R*T_a)*p_ai*1e5)
      rho_m = 1e-2*(((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + rho_o*R*T_w*m_gt*1e3)) 
      w_pc = C_pc*math.sqrt(max(0, rho_m*1e2*(p_wh*1e5 - p_m*1e5)))
      w_pg = (m_gt*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      w_po = (m_ot*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      p_wi = 1e-5*((p_wh*1e5 + g/(A_w*L_w)*max(0,(m_ot*1e3+m_gt*1e3-self.rho_o*L_bh*A_bh))*H_w + 128*mu_oil*L_w*w_pc/(3.14*(D_w**4)*((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*T_w*m_gt*1e3))))
      p_bh = 1e-5*(p_wi*1e5 + self.rho_o*g*H_bh + 128*mu_oil*L_bh*w_po/(math.pi*(D_bh**4)*self.rho_o))
      w_iv = C_iv*math.sqrt(max(0,rho_ai*1e2*(p_ai*1e5 - p_wi*1e5)))
      w_ro = (self.PI)*1e-6*(p_res*1e5 - p_bh*1e5)
      w_rg = 1e1*self.GOR*w_ro

      return w_pg,w_po,w_iv,w_ro,w_rg

    def odecallback(self, t, x, u):
      w_gl = float(u) # current inputs
      # Flows
      w_pg,w_po,w_iv,w_ro,w_rg=self.calculate_flows(x)
      # Process model
      df = [
          (w_gl - w_iv)*1e-3,
          (w_iv - w_pg + w_rg*1e-1)*1e-3,
          (w_ro - w_po)*1e-3
          ]
      return df

    def simulate(self, u):
      self.sim_results = []
      t = [self.stoptime * float(i) / (self.numpoints - 1)
            for i in range(self.numpoints)]
      xnew = u        
      return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))

    def solve_steady_state(self, u):  
      def fobj(x):
        return self.odecallback(0, x, u)
      return fsolve(func=fobj, x0=self.y0)


class GasLiftwedWellSystem:
  def __init__(self, config, costs=[1.0, 0.3]):
    self.oil_cost, self.gas_cost = costs
    self.n_w = len(config.keys())
    self.wells = [GasLiftedWell(k, v.values()) for k,v in config.items()]

  def get_objective(self, u, noise=None):
    fx = 0
    for i, well in enumerate(self.wells):
      x_ss = well.solve_steady_state(u[i])
      _,w_po,_,_,_= well.calculate_flows(x_ss)
      # fx += self.oil_cost * w_po - self.gas_cost * u[i]
      fx += w_po
    
    fx = fx ** 2 - 0.5 * np.sum(u ** 2)

    return fx if noise == None else fx * (1 + np.random.normal(scale=noise))

  
  def get_constraints(self, u, noise=None):
    g = 0
    for i, well in enumerate(self.wells):
      x_ss = well.solve_steady_state(u[i])
      w_pg,_,_,_,_= well.calculate_flows(x_ss)
      g += w_pg
    
    if(noise != None):
      g = g * (1 + np.random.normal(scale=noise))
    
    return np.array([g])


config = { 'well1': { 'GOR': 0.1, 'PI': 2.2, 'rho_o': 900 },
           'well2': { 'GOR': 0.15, 'PI': 2.2, 'rho_o': 800}} 
gmax = 8.0
gs = GasLiftwedWellSystem(config)

fig, ax = plt.subplots(2, 1, figsize=(8,6))

u1 = np.linspace(0.5, 5, 50)
u2 = np.linspace(0.5, 5, 50)

xx, yy = np.meshgrid(u1, u2)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
# make predictions for the grid
cost = np.array([gs.get_objective(x, 0.01) for x in grid])
g = np.array([gs.get_constraints(x, 0.01)[0] for x in grid])
# g = np.array([np.any(gs.get_constraints(x) < gmax) for x in grid])
# reshape the predictions back into a grid
zz_cost = cost.reshape(xx.shape)
zz_g = g.reshape(xx.shape)

CS = ax[0].contourf(xx, yy, zz_cost)
cbar = fig.colorbar(CS, ax=ax[0])
CS = ax[1].contourf(xx, yy, zz_g, cmap='jet')
cbar = fig.colorbar(CS, ax=ax[1])
# gl = GasLiftedWell('test', 0)
# # sim_results = gl.simulate(1)
# ss = gl.get_objective([1])
# print(ss)

#gl.plot_simulation(sim_results)
# fig, ax = plt.subplots(3, 1, figsize=(12,8))
# for u in np.linspace(0.5, 5, 10):
#   sim_results = gl.simulate(u)
#   for i, result in enumerate(sim_results.y):
#     ax[i].plot(sim_results.t, sim_results.y[i])
  
# fig.legend(np.linspace(0.5, 5, 10))
  
  
