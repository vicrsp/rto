import numpy as np

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform

n_iterations = 120
pop_size = 20
max_gen = 100
data_size = 5

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.6139787, 0.00110823, 227.6375114]
u_0 = [10, 0.0005, 240]
g_plant = np.array([0.025, 0.15])

initial_data = generate_samples_uniform(model, plant, g_plant, u_0, data_size)

opt_problem = BatchProfileOptimizer(
    ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15])

adaptation = MAGaussianProcesses(model, initial_data)

rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations)
rto.run(u_0)
