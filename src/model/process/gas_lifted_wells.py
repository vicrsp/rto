import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import casadi as cd
from bunch import Bunch


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
PI = 2.2 # [kg/(s*bar)]
T_a = 28 + 273 # [ºK]
T_w = 32 + 273 # [ºK]
Mw = 20e-3 # [kg]
GOR = 0.1 # [kg/kg]
mu_oil = 1*0.001 # 1cP


class GasLiftedWell:
    def __init__(self, name, k=(2.2, 0.1, 900, 150), y0=[1, 1, 1]):
      self.name = name
      self.y0 = y0
      self.PI, self.GOR, self.rho_o, self.p_res = k
      self.stoptime = 1000
      self.numpoints = 100

    def calculate_flows(self, x):
      m_ga, m_gt, m_ot = x # current state
      
      # p_ai = 1e-5*(((R*T_a/(V_a*Mw) + g*H_a/(L_a*A_a))*m_ga*1e3))
      # p_wh = 1e-5*(((R*T_w/Mw)*(m_gt*1e3/(L_w*A_w + L_r*A_r - m_ot*1e3/self.rho_o))))
      # rho_ai = 1e-2*((Mw/(R*T_a))*p_ai*1e5)
      # # rho_w = 1e-2*(max(0, (m_gt*1e3 + m_ot*1e3 - self.rho_o*L_r*A_r)/(L_w*A_w)))
      # rho_w = 1e-2*(((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*T_w*m_gt*1e3)) 
      # w_pc = C_pc*math.sqrt(max(0, rho_w*1e2*(p_wh*1e5 - p_m*1e5)))
      # w_pg = (m_gt*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      # w_po = (m_ot*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      # p_wi = 1e-5*((p_wh*1e5 + (g/(A_w*L_w))*max(0,(m_ot*1e3+m_gt*1e3-self.rho_o*L_r*A_r))*H_w))
      # p_bh = 1e-5*(p_wi*1e5 + rho_w*1e2*g*H_r)
      # w_iv = C_iv*math.sqrt(max(0,rho_ai*1e2*(p_ai*1e5 - p_wi*1e5)))
      # w_ro = (self.PI)*1e-6*(self.p_res*1e5 - p_bh*1e5)
      # w_rg = 1e1*self.GOR*w_ro

      p_ai = 1e-5*(((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3) + (Mw/(R*T_a)*((R*T_a/(V_a*Mw) + g*H_a/V_a)*m_ga*1e3))*g*H_a)
      p_wh = 1e-5*(((R*T_w/Mw)*(m_gt*1e3/(L_w*A_w + L_bh*A_bh - m_ot*1e3/self.rho_o))) - ((m_gt*1e3+m_ot*1e3 )/(L_w*A_w))*g*H_w/2)
      rho_ai = 1e-2*(Mw/(R*T_a)*p_ai*1e5)
      rho_m = 1e-2*(((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*T_w*m_gt*1e3)) 
      w_pc = C_pc*math.sqrt(max(0, rho_m*1e2*(p_wh*1e5 - p_m*1e5)))
      w_pg = (m_gt*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      w_po = (m_ot*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
      p_wi = 1e-5*((p_wh*1e5 + g/(A_w*L_w)*max(0,(m_ot*1e3+m_gt*1e3-self.rho_o*L_bh*A_bh))*H_w + 128*mu_oil*L_w*w_pc/(math.pi*(D_w**4)*((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*T_w*m_gt*1e3))))
      p_bh = 1e-5*(p_wi*1e5 + self.rho_o*g*H_bh + 128*mu_oil*L_bh*w_po/(math.pi*(D_bh**4)*self.rho_o))
      w_iv = C_iv*math.sqrt(max(0,rho_ai*1e2*(p_ai*1e5 - p_wi*1e5)))
      w_ro = (self.PI)*1e-6*(self.p_res*1e5 - p_bh*1e5)
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

    self.L_w = np.array([1500] * self.n_w) # [m]
    self.H_w = np.array([1000] * self.n_w) # [m]
    self.D_w = np.array([0.121] * self.n_w) # [m]

    self.L_bh = np.array([500] * self.n_w) # [m]
    self.H_bh = np.array([100] * self.n_w) # [m]
    self.D_bh = np.array([0.121] * self.n_w) # [m]

    self.L_a = np.array([1500] * self.n_w) # [m]
    self.H_a = np.array([1000] * self.n_w) # [m]
    self.D_a = np.array([0.189] * self.n_w) # [m]
    self.A_a = math.pi * (self.D_a / 2)**2 # [m2]

    self.A_w = math.pi*((self.D_w/2)**2)
    self.A_bh = math.pi*((self.D_bh/2)**2)
    self.V_a = self.L_a * (math.pi*(self.D_a / 2)**2 - math.pi*(self.D_w / 2)**2)

    self.rho_o = np.array([900] * self.n_w) # [kg/m3]
    self.C_iv = np.array([1e-4] * self.n_w) # [m2]
    self.C_pc = np.array([1e-3] * self.n_w) # [m2]
    self.p_m = np.array([20] * self.n_w) # [bar]
    self.PI = np.array([2.2] * self.n_w) # [kg/(s*bar)]
    self.T_a = np.array([28 + 273] * self.n_w) # [ºK]
    self.T_w = np.array([32 + 273] * self.n_w) # [ºK]
    self.Mw = np.array([20e-3] * self.n_w) # [kg]
    self.GOR = np.array([0.1] * self.n_w) # [kg/kg]
    self.mu_oil = np.array([1*0.001] * self.n_w) # 1cP
    self.p_res = np.array([150] * self.n_w)

    self.numpoints = 500

  def build_casadi_sys(self):
    # differential states
    m_ga = cd.MX.sym('m_ga',self.n_w) # 1-2
    m_gt = cd.MX.sym('m_gt',self.n_w) # 3-4
    m_ot = cd.MX.sym('m_ot',self.n_w) # 5-6

    # control input
    w_gl = cd.MX.sym('w_gl', self.n_w)

    # algebraic equations used for substitution in the ODE model
    p_ai = 1e-5*(((R*self.T_a/(self.V_a*Mw) + g*self.H_a/self.V_a)*m_ga*1e3) + (Mw/(R*self.T_a)*((R*self.T_a/(self.V_a*Mw) + g*self.H_a/self.V_a)*m_ga*1e3))*g*self.H_a)
    p_wh = 1e-5*(((R*self.T_w/Mw)*(m_gt*1e3/(self.L_w*self.A_w + self.L_bh*self.A_bh - m_ot*1e3/self.rho_o))) - ((m_gt*1e3+m_ot*1e3 )/(self.L_w*self.A_w))*g*self.H_w/2)
    rho_ai = 1e-2*(Mw/(R*self.T_a)*p_ai*1e5)
    rho_m = 1e-2*(((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*self.T_w*m_gt*1e3)) 
    w_pc = self.C_pc*cd.sqrt(cd.fmax(0, rho_m*1e2*(p_wh*1e5 - self.p_m*1e5)))
    w_pg = (m_gt*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
    w_po = (m_ot*1e3/(m_gt*1e3+m_ot*1e3))*w_pc
    p_wi = 1e-5*((p_wh*1e5 + g/(self.A_w*self.L_w)*cd.fmax(0, m_ot*1e3+m_gt*1e3-self.rho_o*self.L_bh*self.A_bh)*self.H_w + 128*self.mu_oil*self.L_w*w_pc/(3.14*self.D_w**4*((m_gt*1e3 + m_ot*1e3)*p_wh*1e5*Mw*self.rho_o)/(m_ot*1e3*p_wh*1e5*Mw + self.rho_o*R*self.T_w*m_gt*1e3))))
    p_bh = 1e-5*(p_wi*1e5 + self.rho_o*g*self.H_bh + 128*self.mu_oil*self.L_bh*w_po/(3.14*self.D_bh**4*self.rho_o))
    w_iv = self.C_iv*cd.sqrt(rho_ai*1e2*(p_ai*1e5 - p_wi*1e5))
    w_ro = (self.PI)*1e-6*(self.p_res*1e5 - p_bh*1e5)
    w_rg = 1e1*self.GOR*w_ro

    # differential equations
    df1C = (w_gl - w_iv)*1e-3
    df2C = (w_iv + w_rg*1e-1 - w_pg)*1e-3
    df3C = (w_ro - w_po)*1e-3

    # Concatenate the differential and algebraic equations
    diff = cd.vertcat(df1C,df2C,df3C)

    # concatenate the differential and algebraic states
    x_var = cd.vertcat(m_ga,m_gt,m_ot)
    u_var = cd.vertcat(w_gl)

    z_var = cd.vertcat(p_ai,p_wh,p_wi,p_bh,rho_ai,rho_m,w_iv,w_pc,w_pg,w_po,w_ro,w_rg)
    algebraic_eq = cd.Function('alg',[x_var,u_var],[z_var])
    y_var = cd.vertcat(w_pg,w_po,w_iv,w_ro,w_rg)
    y_model= cd.Function('alg',[x_var,u_var],[y_var])

    # cost function
    L_linear = -self.oil_cost*cd.sum1(w_po) + self.gas_cost*cd.sum1(w_gl)
    L_quad = cd.power(cd.sum1(w_po), 2) - cd.sum1(cd.power(w_gl, 2))
    constraint = cd.sum1(w_pg)

    ode = {'x': x_var,'p':u_var,'ode': diff, 'quad': L_linear}
    
    f_linear = cd.Function('f_linear',[x_var,u_var],[L_linear])
    f_quad = cd.Function('f_quad',[x_var,u_var],[L_quad])
    f_g = cd.Function('g',[x_var,u_var],[constraint])

    sys_dict = {}
    sys_dict['x']= x_var
    sys_dict['z'] = []
    sys_dict['u'] = u_var
    sys_dict['d'] = []
    sys_dict['diff'] = diff
    sys_dict['alg'] = algebraic_eq
    sys_dict['y'] = y_var
    sys_dict['y_model'] = y_model

    sys_dict['nlcon'] = []
    sys_dict['lb'] =  []
    sys_dict['ub'] = []
    sys_dict['ode'] = ode

    sys_dict['f_linear'] = f_linear
    sys_dict['f_quad'] = f_quad
    sys_dict['f_g'] = f_g

    return sys_dict

  def simulate_casadi(self, x0, u, tf=1.0):
    sys = self.build_casadi_sys()
    ode = sys['ode']
    # create IDAS integrator
    opts = {'tf': tf}
    F = cd.integrator('F','cvodes', ode, opts)
    # Integrate step by step
    x = x0
    results = []
    results_well = {}
    for _ in range(self.numpoints):
      res = F(x0=x,p=u)
      x = res["xf"]
      y = sys['y_model'](x,u)
      sim = np.concatenate((x, y))
      results.append(sim.reshape(-1,))

    results = np.array(results)
    for i, well in enumerate(self.wells):
      results_well[well.name] = Bunch(y=results[:, i::self.n_w].T, t= np.arange(results.shape[0]))

    return results_well
  
  def simulate(self, u):
    results = {}
    for i, well in enumerate(self.wells):
      sim = well.simulate(u[i])
      m = sim.y.shape[1]
      flows = [well.calculate_flows(sim.y[:,k]) for k in range(m)]
      sim.y = np.concatenate((sim.y, np.array(flows).T))
      results[well.name] = sim

    return results


  def get_steady_state_casadi(self, u):
    results = {}
    xss, y = self.solve_steady_state_casadi(u)
    xss = np.concatenate((xss, y))

    for i, well in enumerate(self.wells):
      results[well.name] = xss[i::self.n_w]

    return results

  def solve_steady_state_casadi(self, u):
    
    sys = self.build_casadi_sys()

    dx0 = np.array([1.32*np.ones((self.n_w,)), 0.8*np.ones((self.n_w,)), 6*np.ones((self.n_w,))]).flatten()
    lbx = np.array([0.01*np.ones((self.n_w,)), 0.01*np.ones((self.n_w,)), 0.01*np.ones((self.n_w,))]).flatten()
    ubx = np.array([1e7*np.ones((self.n_w,)), 1e7*np.ones((self.n_w,)), 1e7*np.ones((self.n_w,))]).flatten()
   
    lbw = np.concatenate((lbx,u))
    ubw = np.concatenate((ubx,u))
    w0 = np.concatenate((dx0,u))

    g = cd.vertcat(sys['diff'])
    lbg = np.zeros((sys['diff'].shape[0],))
    ubg = np.zeros((sys['diff'].shape[0],))


    nlp = {'x': cd.vertcat(sys['x'], sys['u']),'p': sys['d'],'f': 0,'g': g}
    opts = {'warn_initial_bounds': False, 'print_time': False,'ipopt': {'print_level':0}}
    solver = cd.nlpsol('solver','ipopt', nlp, opts)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw,lbg=lbg,ubg=ubg)

    y = sys['y_model'](sol['x'][:-self.n_w],u)
    return np.array(sol['x']), y

  def get_objective(self, u, noise=None):
    return self.get_objective_quadratic(u, noise)

  def get_objective_quadratic(self, u, noise=None):
    sys = self.build_casadi_sys()
    xss, _ = self.solve_steady_state_casadi(u)

    fx = float(sys['f_quad'](xss[:-2], u))

    return fx if noise == None else fx * (1 + np.random.normal(scale=noise))

  def get_objective_linear(self, u, noise=None):
    sys = self.build_casadi_sys()
    xss, _ = self.solve_steady_state_casadi(u)

    fx = float(sys['f_linear'](xss[:-2], u))
    return -fx if noise == None else -fx * (1 + np.random.normal(scale=noise))

  def get_constraints(self, u, noise=None):
    sys = self.build_casadi_sys()
    xss, _ = self.solve_steady_state_casadi(u)

    g = float(sys['f_g'](xss[:-2], u))
    
    if(noise != None):
      g = g * (1 + np.random.normal(scale=noise))
    
    return np.array([g])

    
  def get_steady_state(self, u):
    results = {}
    for i, well in enumerate(self.wells):
      xss = well.solve_steady_state(u[i])
      w_pg,w_po,w_iv,w_ro,w_rg = well.calculate_flows(xss)
      #print(xss.shape)
      xss = np.concatenate((xss, [w_pg,w_po,w_iv,w_ro,w_rg]))
      results[well.name] = xss

    return results
    #return {well.name: well.solve_steady_state(u[i]) for i, well in enumerate(self.wells)}


# config = { 'well1': { 'GOR': 0.1, 'PI': 2.2, 'rho_o': 900, 'p_res': 150 },
#           'well2': { 'GOR': 0.1, 'PI': 2.2, 'rho_o': 900, 'p_res': 150 }} 
# gs = GasLiftwedWellSystem(config)
# # gs.casadi_ode()
# sim_res = gs.simulate_casadi(x0=[1,1,1,1,1,1],u=[1,1],tf=1)
# sol = gs.get_steady_state_casadi(u=np.ones((gs.n_w,)))
# fquad = gs.get_objective_quadratic(u=[1,1])
# fl = gs.get_objective_linear(u=[1,1])
# fg = gs.get_constraints(u=[1,1])


# u1 = np.linspace(0.5, 5.0, 10)
# u2 = np.linspace(0.5, 5.0, 10)

# xx, yy = np.meshgrid(u1, u2)
# # flatten each grid to a vector
# r1, r2 = xx.flatten(), yy.flatten()
# r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# # horizontal stack vectors to create x1,x2 input for the model
# grid = np.hstack((r1,r2))
# # make predictions for the grid
# cost_quad = np.array([gs.get_objective_quadratic(x,0.1) for x in grid])
# cost_linear = np.array([gs.get_objective_linear(x,0.1) for x in grid])
# g = np.array([gs.get_constraints(x,0.1)[0] for x in grid])