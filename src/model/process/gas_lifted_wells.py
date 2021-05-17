import numpy as np
from gekko import GEKKO
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# CONSTANTS
R = 8.31446261815324 # gas constant
g = 9.80665 # [m/s2]

Lw = 1500 # [m]
Hw = 1000 # [m]
Dw = 0.121 # [m]
Aw = math.pi * (Dw / 2)**2 # [m2]

Lr = 500 # [m]
Hr = 100 # [m]
Dr = 0.121 # [m]
Ar = math.pi * (Dr / 2)**2 # [m2]

La = 1500 # [m]
Ha = 1000 # [m]
Da = 0.189 # [m]
Aa = math.pi * (Da / 2)**2 # [m2]
Va = La * (math.pi*(Da / 2)**2 - math.pi*(Dw / 2)**2)

rho_o = 900 # [kg/m3]
C_iv = 1e-4 # [m2]
C_pc = 1e-3 # [m2]
p_m = 20 # [bar]
p_r = 150 # [bar]
PI = 2.2 # [kg/s/bar]
Ta = 28 + 273 # [ºK]
Tw = 32 + 273 # [ºK]
Mw = 20 # [g]
GOR = 0.1 # [kg/kg]


class GasLiftedWell:
    def __init__(self, n_wells, g, y0=[1, 1, 1]):
      self.n_wells = n_wells
      self.g = g
      self.y0 = y0
      self.stoptime = 300 * 60
      self.numpoints = 100
      self.sim_results = []
      
    def odecallback(self, t, w, x):
      m_ga, m_gt, m_ot = w
      w_gl = x

      # Intermediates
      p_a = 1e-5*((m_ga*1e3) * ((Ta*R/(Va*Mw) + g*La/(Aa*La))))
      p_wh = 1e-5*((Tw*R/Mw)*(m_gt*1e3)/(Lw*Aw + Lr*Ar - ((m_ot*1e3)/rho_o)))
      p_wi = 1e-5*(p_wh*1e5 + (g/(Aw*Lw))*max(0,m_ot*1e3 + m_gt*1e3 - rho_o*Lr*Ar)*Hw)
      p_bh = 1e-5*(p_wi*1e5 + rho_o*g*Hr)
      
      rho_a = 1e-2*(Mw * (p_a*1e5) / (Ta * R))
      rho_m = 1e-2*(m_gt*1e3 + m_ot*1e3 - rho_o * Lr * Ar)/(Lw*Aw)
      
      w_pc = C_pc*np.sqrt((rho_m*1e2)*max(0, p_wh*1e5 - p_m*1e5))
      w_ro = (PI*1e-6)*(p_r*1e5 - p_bh*1e5)
      w_rg = 1e1*GOR*w_ro
      w_iv = C_iv*np.sqrt((rho_a*1e2)*max(0, p_a*1e5 - p_wi*1e5))

      w_pg = (m_gt*1e3)*w_pc/(m_gt*1e3 + m_ot*1e3)
      w_po = (m_ot*1e3)*w_pc/(m_gt*1e3 + m_ot*1e3)

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

    def plot_simulation(self, sim_results):
      for i, result in enumerate(sim_results.y):
        plt.figure(figsize=(8,6))
        plt.plot(sim_results.t, result)
        plt.title(f'Signal {i}')
        
        
gl = GasLiftedWell(1, [])
sim = gl.simulate(2)
gl.plot_simulation(sim)
