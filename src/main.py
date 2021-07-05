import numpy as np
import multiprocessing
import yaml

from rto.rto import RTO
from rto.optimization.optimizer import ModelBasedOptimizer
from rto.adaptation.ma_gaussian_processes import MAGaussianProcesses
from rto.models.semi_batch import SemiBatchReactor
from rto.utils import generate_samples_uniform

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103]
g_plant = np.array([0.025, 0.15])
x_ub = [30, 0.002, 250]
x_lb = [0, 0, 200]

def run_rto(n_experiments, n_iterations, data_array, solver, db_file, neighbors, exp_name, noise, backoff):
    for i in range(n_experiments):
        print('{} experiment {}'.format(exp_name, i))
        initial_data = data_array[i]
        # build the model-based optimization problem
        opt_problem = ModelBasedOptimizer(
            x_ub, x_lb, g_plant, solver=solver, backoff=backoff)

        # build the adaptation model
        adaptation = MAGaussianProcesses(
            model, initial_data, filter_data=True, neighbors_type=neighbors)

        u_0_feas = initial_data[0][-1]
        rto = RTO(model, plant, opt_problem, adaptation,
                  iterations=n_iterations, db_file=db_file, name=exp_name, noise=noise)

        rto.run(u_0_feas)


# loads the experiment config
with open('/home/victor/git/rto/src/rto/experiment/configs/rto_magp.yaml') as f:
    config = yaml.safe_load(f)

db_file = config['db_file']
adaptation = config['adaptation']
repetitions = config['repetitions']
rto_iterations = config['rto_iterations']
initial_data_size = config['initial_data_size']
initial_data_noise = config['initial_data_noise']

configs = config['configs']

# generate all the data used in the experiments
initial_data_array = [generate_samples_uniform(
    model, plant, g_plant, u_0, initial_data_size, noise=initial_data_noise) for _ in range(repetitions)]

# create the list of jobs to be run
jobs = []
for key, cfg in configs.items():
    solver = cfg['solver']
    neighbors = cfg['neighbors']
    noise = cfg['noise']
    backoff = cfg['backoff']
    name = cfg['name']

    exp_name = f'{adaptation}-{name}-{neighbors}-{noise}-{backoff}'

    p = multiprocessing.Process(target=run_rto, args=(
        repetitions, rto_iterations, initial_data_array, solver, db_file, neighbors, exp_name, noise, backoff))
    p.start()
    jobs.append(p)

# wait for all to finish
[job.join() for job in jobs]
