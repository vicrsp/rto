import numpy as np

from optimization.rto import RTO
from model.optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from optimization.de import DifferentialEvolution


pop_size = 20
max_gen = 100
de_type = 'rand/1/bin'
data_size = 30

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
model_ideal = SemiBatchReactor()
u_real_optimum = [18.6139787, 0.00110823, 227.6375114]

u_initial = []
y_initial = []
for _ in range(data_size):
    u_rand = u_real_optimum * (1 + np.random.randn(len(u_real_optimum)) * 0.1)
    sim_ideal = model_ideal.simulate(u_rand)
    sim_model = model.simulate(u_rand)

    fr, gr = model_ideal.get_objective(
        sim_ideal), model_ideal.get_constraints(u_rand, sim_ideal)

    fm, gm = model.get_objective(
        sim_model), model.get_constraints(u_rand, sim_model)

    u_initial.append(u_rand)
    y_initial.append(np.append(fr - fm, gr - gm))


opt_problem = BatchProfileOptimizer(
    ub=[250, 0.002, 250], lb=[0, 0, 0], g=[0.025, 0.15, 0])

optimizer = DifferentialEvolution(lb=opt_problem.lb, ub=opt_problem.ub,
                                  callback=opt_problem.save_results, max_generations=max_gen, pop_size=pop_size, de_type=de_type)

opt_problem.set_optimizer(optimizer)

adaptation = MAGaussianProcesses(model, [np.asarray(u_initial), np.asarray(y_initial)])

rto = RTO(model, model_ideal, opt_problem, adaptation)
rto.run()
