import numpy as np
import multiprocessing

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform

n_experiments = 10
n_iterations = 60
data_size = 10

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.4427644, 0.00110823777, 227.792418]
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103]
g_plant = np.array([0.025, 0.15])
x_ub = [30, 0.002, 250]
x_lb = [0, 0, 200]


def run_rto(n_experiments, data_array, solver, db_file, neighbors, exp_name):
    for i in range(n_experiments):
        print('{} experiment {}'.format(exp_name, i))
        initial_data = data_array[i]
        # build the model-based optimization problem
        opt_problem = BatchProfileOptimizer(
            x_ub, x_lb, g_plant, solver=solver)

        # build the adaptation model
        adaptation = MAGaussianProcesses(
            model, initial_data, filter_data=True, neighbors_type=neighbors)

        u_0_feas = initial_data[0][-1]
        rto = RTO(model, plant, opt_problem, adaptation,
                  iterations=n_iterations, db_file=db_file, name=exp_name)

        rto.run(u_0_feas)


def run_rto_experiment(n_experiments, initial_data_size, neighbors, suffix, config, max_threads=4):

    # generate all the data before
    initial_data_array = [generate_samples_uniform(
        model, plant, g_plant, u_0, initial_data_size) for i in range(n_experiments)]

    # create the list of jobs to be run
    jobs = []

    for cfg in config:
        solver = cfg['solver']
        db_file = cfg['db_file']
        exp_name = 'ma-gp-{}-{}-{}'.format(solver, neighbors, suffix)

        p = multiprocessing.Process(target=run_rto, args=(
            n_experiments, initial_data_array, solver, db_file, neighbors, exp_name))
        p.start()
        jobs.append(p)

    # wait for all to finish
    [job.join() for job in jobs]


config = [{'solver': 'slsqp_scipy',
           'db_file': '/mnt/d/rto_data/rto_poc_exact_slsqp.db'},
          {'solver': 'de_scipy_best1bin',
           'db_file': '/mnt/d/rto_data/rto_poc_debest1bin.db'}]


run_rto_experiment(n_experiments, data_size, 'k_last', 'same_samples', config)
# def run_rto_exact(neighbors, suffix=''):
#     exp_name = 'ma-gp-slsqp-{}-{}'.format(neighbors, suffix)
#     # Exact algorithm
#     for i in range(n_experiments):
#         print('{} experiment {}'.format(exp_name, i))
#         initial_data = generate_samples_uniform(
#             model, plant, g_plant, u_0, data_size)
#         opt_problem = BatchProfileOptimizer(
#             ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='slsqp_scipy')

#         adaptation = MAGaussianProcesses(model, initial_data, filter_data=True)
#         u_0_feas = initial_data[0][-1]
#         rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
#                   db_file='/mnt/d/rto_data/rto_poc_exact_slsqp.db', name=exp_name)
#         rto.run(u_0_feas)

# def run_rto_de_best1bin(neighbors, suffix=''):
#     # DE best1bin algorithm
#     exp_name = 'ma-gp-best1bin-{}-{}'.format(neighbors, suffix)
#     for i in range(n_experiments):
#         print('{} experiment {}'.format(exp_name, i))
#         initial_data = generate_samples_uniform(
#             model, plant, g_plant, u_0, data_size)
#         opt_problem = BatchProfileOptimizer(
#             ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_best1bin')

#         adaptation = MAGaussianProcesses(model, initial_data, neighbors, filter_data=True)

#         rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
#                   db_file='/mnt/d/rto_data/rto_poc_debest1bin.db', name=exp_name)
#         rto.run(u_0)


# def run_rto_de_rand1bin(neighbors):
#     # DE rand1bin algorithm
#     exp_name = 'ma-gp-rand1bin-{}'.format(neighbors)
#     for i in range(n_experiments):
#         print('{} experiment {}'.format(exp_name, i))
#         initial_data = generate_samples_uniform(
#             model, plant, g_plant, u_0, data_size)
#         opt_problem = BatchProfileOptimizer(
#             ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_rand1bin')

#         adaptation = MAGaussianProcesses(model, initial_data, neighbors, filter_data=True)

#         rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
#                   db_file='/mnt/d/rto_data/rto_poc_derand1bin.db', name=exp_name)
#         rto.run(u_0)


# def run_rto_experiment(n_experiments, neighbors, suffix, solvers=['de_scipy_best1bin','slsqp_scipy']):
#     for i in range(n_experiments):

# Define general constants
#p1 = multiprocessing.Process(target=run_rto_exact, args=('k_last', 'noiseless'))
# p1.start()
#p2 = multiprocessing.Process(target=run_rto_de_best1bin, args=('k_last', '5percNoise'))
# p2.start()
#p3 = multiprocessing.Process(target=run_rto_de_rand1bin, args=('k_nearest',))
# p3.start()
#jobs = [p1]
#[job.join() for job in jobs]
