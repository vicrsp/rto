import numpy as np

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor

n_cycles = 120
pop_size = 20
max_gen = 100
de_type = 'rand/1/bin'
data_size = 100

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
model_ideal = SemiBatchReactor()
u_real_optimum = [18.6139787, 0.00110823, 227.6375114]
u_0 = u_real_optimum #[15, 0.0015, 230]
g_plant = np.array([0.025, 0.15])

u_initial = []
y_initial = []
for _ in range(data_size):
    u_rand = u_0 * (1 + np.random.randn(len(u_0)) * 0.5)
    sim_ideal = model_ideal.simulate(u_rand)
    sim_model = model.simulate(u_rand)

    fr, gr = model_ideal.get_objective(
        sim_ideal), model_ideal.get_constraints(u_rand, sim_ideal)

    fm, gm = model.get_objective(
        sim_model), model.get_constraints(u_rand, sim_model)

    if(np.any(gr - g_plant > 0) == False):
        u_initial.append(u_rand)
        y_initial.append(np.append(fr - fm, gr - gm))


opt_problem = BatchProfileOptimizer(
    ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15])

adaptation = MAGaussianProcesses(model, [np.asarray(u_initial), np.asarray(y_initial)])

rto = RTO(model, model_ideal, opt_problem, adaptation, cycles=n_cycles)
rto.run(u_0)
