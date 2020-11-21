import numpy as np
import multiprocessing

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform

n_experiments = 20
n_iterations = 60
data_size = 10

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.4427644, 0.00110823777, 227.792418]
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103]
g_plant = np.array([0.025, 0.15])


def run_rto_exact():
    # Exact algorithm
    initial_data = generate_samples_uniform(
        model, plant, g_plant, u_0, data_size)
    opt_problem = BatchProfileOptimizer(
        ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='slsqp_scipy')

    adaptation = MAGaussianProcesses(
        model, initial_data, 'k_nearest', filter_data=True)
    u_0_feas = initial_data[0][0]
    rto = RTO(model, plant, opt_problem, adaptation,
              iterations=n_iterations, name='ma-gp-slsqp_scipy')
    rto.run(u_0_feas)


def run_rto_de_best1bin():
    # DE best1bin algorithm
    initial_data = generate_samples_uniform(
        model, plant, g_plant, u_0, data_size, offset=0.5)
    opt_problem = BatchProfileOptimizer(
        ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_best1bin')

    adaptation = MAGaussianProcesses(
        model, initial_data, 'k_nearest', filter_data=True)

    rto = RTO(model, plant, opt_problem, adaptation,
              iterations=n_iterations, name='ma-gp-de_scipy_best1bin')
    rto.run(u_0)


def run_rto_de_rand1bin():
    initial_data = generate_samples_uniform(
        model, plant, g_plant, u_0, data_size)
    opt_problem = BatchProfileOptimizer(
        ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_rand1bin')

    adaptation = MAGaussianProcesses(model, initial_data)

    rto = RTO(model, plant, opt_problem, adaptation,
              iterations=n_iterations, name='ma-gp-de_scipy_rand1bin')
    rto.run(u_0)


if __name__ == '__main__':
    run_rto_exact()
    #run_rto_de_best1bin()
