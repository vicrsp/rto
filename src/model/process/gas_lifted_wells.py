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
    def __init__(self, y0=[1, 1, 1]):
      self.y0 = y0
      self.stoptime = 300 * 60
      self.numpoints = 100
      self.sim_results = []
      
    def odecallback(self, t, x, u):
      m_ga, m_gt, m_ot = x # current state
      w_gl = u[0] # current inputs

      # Intermediates
      # p_a = 1e-5*((m_ga*1e3) * ((Ta*R/(Va*Mw) + g*La/(Aa*La))))
      # p_wh = 1e-5*((Tw*R/Mw)*(m_gt*1e3)/(Lw*Aw + Lr*Ar - ((m_ot*1e3)/rho_o)))
      # p_wi = 1e-5*(p_wh*1e5 + (g/(Aw*Lw))*max(0,m_ot*1e3 + m_gt*1e3 - rho_o*Lr*Ar)*Hw)
      # p_bh = 1e-5*(p_wi*1e5 + rho_o*g*Hr)
      
      # rho_a = 1e-2*(Mw * (p_a*1e5) / (Ta * R))
      # rho_m = 1e-2*(m_gt*1e3 + m_ot*1e3 - rho_o * Lr * Ar)/(Lw*Aw)
      
      # w_pc = C_pc*np.sqrt((rho_m*1e2)*max(0, p_wh*1e5 - p_m*1e5))
      # w_ro = (PI*1e-6)*(p_r*1e5 - p_bh*1e5)
      # w_rg = 1e1*GOR*w_ro
      # w_iv = C_iv*np.sqrt((rho_a*1e2)*max(0, p_a*1e5 - p_wi*1e5))

      # w_pg = (m_gt*1e3)*w_pc/(m_gt*1e3 + m_ot*1e3)
      # w_po = (m_ot*1e3)*w_pc/(m_gt*1e3 + m_ot*1e3)
      p_ai = 1e-5*(((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3) + (Mw/(R*T_a)*((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3))*g*H_a)
      p_wh = 1e-5*(((R*T_w/Mw)*(m_gt*1e3/(L_w*A_w + L_bh*A_bh - m_ot*1e3/rho_o))) - ((m_gt*1e3+m_ot*1e3 )/(L_w*A_w))*g*H_w/2)
      rho_ai = 1e-2*(Mw/(R*T_a)*p_ai*1e5)
      rho_m = 1e-2*(((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*rho_o)/(m_ot*1e3*p_wh*1e5*Mw + rho_o*R*T_w*m_gt*1e3)) 
      w_pc = C_pc*math.sqrt(max(0, rho_m*1e2*(p_wh*1e5 - p_m*1e5)))
      w_pg = (m_gt*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      w_po = (m_ot*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      p_wi = 1e-5*((p_wh*1e5 + g/(A_w*L_w)*max(0,(m_ot*1e3+m_gt*1e3-rho_o*L_bh*A_bh))*H_w + 128*mu_oil*L_w*w_pc/(3.14*(D_w**4)*((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*rho_o)/(m_ot*1e3*p_wh*1e5*Mw + rho_o*R*T_w*m_gt*1e3))))
      p_bh = 1e-5*(p_wi*1e5 + rho_o*g*H_bh + 128*mu_oil*L_bh*w_po/(math.pi*(D_bh**4)*rho_o))
      w_iv = C_iv*math.sqrt(max(0,rho_ai*1e2*(p_ai*1e5 - p_wi*1e5)))
      w_ro = (PI)*1e-6*(p_res*1e5 - p_bh*1e5)
      w_rg = 1e1*GOR*w_ro

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
      xnew = [u]         
      return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))

    def solve_steady_state(self, u):  
      def fobj(x):
        return self.odecallback(0, x, u)
      return fsolve(func=fobj, x0=self.y0)
    
    def plot_simulation(self, sim_results):
      for i, result in enumerate(sim_results.y):
        plt.figure(figsize=(8,6))
        plt.plot(sim_results.t, result)
        plt.title(f'Signal {i}')
        
        
gl = GasLiftedWell()
# sim_results = gl.simulate(1)
ss = gl.solve_steady_state([1])
print(ss)

#gl.plot_simulation(sim_results)
# fig, ax = plt.subplots(3, 1, figsize=(12,8))
# for u in np.linspace(0.5, 5, 10):
#   sim_results = gl.simulate(u)
#   for i, result in enumerate(sim_results.y):
#     ax[i].plot(sim_results.t, sim_results.y[i])
  
# fig.legend(np.linspace(0.5, 5, 10))
  
  
