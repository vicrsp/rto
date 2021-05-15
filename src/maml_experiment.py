import numpy as np
import multiprocessing

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_machine_learning import MAMachineLearningRegression
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform
from sklearn.ensemble import RandomForestRegressor

n_experiments = 30
n_iterations = 60
data_size = 5
initial_data_noise = 0.01

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.4427644, 0.00110823777, 227.792418]
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103]
g_plant = np.array([0.025, 0.15])
x_ub = [30, 0.002, 250]
x_lb = [0, 0, 200]


def run_rto(n_experiments, data_array, solver, db_file, neighbors, exp_name, noise, backoff):
    for i in range(n_experiments):
        print('{} experiment {}'.format(exp_name, i))
        initial_data = data_array[i]
        # build the model-based optimization problem
        opt_problem = BatchProfileOptimizer(
            x_ub, x_lb, g_plant, solver=solver, backoff=backoff)

        # build the adaptation model
        adaptation = MAMachineLearningRegression(
            model, initial_data, model=RandomForestRegressor(), filter_data=True, neighbors_type=neighbors)

        u_0_feas = initial_data[0][-1]
        rto = RTO(model, plant, opt_problem, adaptation,
                  iterations=n_iterations, db_file=db_file, name=exp_name, noise=noise)

        rto.run(u_0_feas)


def run_rto_experiment(n_experiments, initial_data_size, initial_data_noise, config, max_threads=4):

    # generate all the data before
    initial_data_array = [generate_samples_uniform(
        model, plant, g_plant, u_0, initial_data_size, noise=initial_data_noise) for i in range(n_experiments)]

    # create the list of jobs to be run
    jobs = []

    for cfg in config:
        solver = cfg['solver']
        db_file = cfg['db_file']
        neighbors = cfg['neighbors']
        noise = cfg['noise']
        backoff = cfg['backoff']

        exp_name = 'ma-rf-{}-{}-{}-{}'.format(solver,
                                              neighbors, noise, backoff)

        p = multiprocessing.Process(target=run_rto, args=(
            n_experiments, initial_data_array, solver, db_file, neighbors, exp_name, noise, backoff))
        p.start()
        jobs.append(p)

    # wait for all to finish
    [job.join() for job in jobs]


# config = [{'solver': 'de_scipy_best1bin',
#            'db_file': '/mnt/d/rto_data/rto_poc_de_experiments.db',
#            'neighbors': 'k_last',
#            'noise': 0.05,
#            'backoff': 0.00}]
# config = [{'solver': 'de_scipy_best1bin',
#            'db_file': '/mnt/d/rto_data/rto_poc_derand1bin.db',
#            'neighbors': 'k_last',
#            'noise': 0.05,
#            'backoff': 0.05},
#            {'solver': 'slsqp_scipy',
#            'db_file': '/mnt/d/rto_data/rto_poc_exact_slsqp.db',
#            'neighbors': 'k_last',
#            'noise': 0.05,
#            'backoff': 0.05},
#           {'solver': 'de_scipy_best1bin',
#            'db_file': '/mnt/d/rto_data/rto_poc_debest1bin.db',
#            'neighbors': 'k_last',
#            'noise': 0.05,
#            'backoff': 0.00}]
config = [{'solver': 'slsqp_scipy',
           'db_file': '/mnt/d/rto_data/rto_poc_sqp_experiments.db',
           'neighbors': 'k_last',
           'noise': 0.01,
           'backoff': 0.0},
          {'solver': 'de_scipy_best1bin',
           'db_file': '/mnt/d/rto_data/rto_poc_de_experiments.db',
           'neighbors': 'k_last',
           'noise': 0.01,
           'backoff': 0.00}]


run_rto_experiment(n_experiments, data_size, initial_data_noise, config)
