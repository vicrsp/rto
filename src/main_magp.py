from optimization.rto import RTO
from model.optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from optimization.de import DifferentialEvolution


pop_size = 20
max_gen = 100
de_type = 'rand/1/bin'

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
model_ideal = SemiBatchReactor()

opt_problem = BatchProfileOptimizer(
    ub=[250, 0.002, 250], lb=[0, 0, 0], g=[0.025, 0.15])

optimizer = DifferentialEvolution(lb=opt_problem.lb, ub=opt_problem.ub,
                                  callback=opt_problem.save_results, max_generations=max_gen, pop_size=pop_size, de_type=de_type)

opt_problem.set_optimizer(optimizer)

adaptation = MAGaussianProcesses(model)

rto = RTO(model, model_ideal, opt_problem, adaptation)
rto.run()