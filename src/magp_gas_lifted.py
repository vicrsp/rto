import multiprocessing

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.gas_lifted_wells import GasLiftwedWellSystem
from model.utils import generate_samples_uniform

n_experiments = 30
n_iterations = 60
data_size = 5
initial_data_noise = 0.01
n_wells = 2

def build_config(n_wells, gor, PI, rho_o):
    return { f'well{i}':  { 'GOR': gor[i], 'PI': PI[i], 'rho_o': rho_o[i] } for i in range(n_wells)}

config_model = build_config(n_wells, [0.1,0.15], [2.2,2.2], [900,800])
config_plant = build_config(n_wells, [0.1,0.15], [2.8,2.6], [900,800])

model = GasLiftwedWellSystem(config_model)
plant = GasLiftwedWellSystem(config_plant)

u_0 = [1.0] * n_wells
g_plant = n_wells * 7
x_ub = [5] * n_wells
x_lb = [0.5] * n_wells

def run_rto(n_experiments, data_array, solver, db_file, neighbors, exp_name, noise, backoff):
    for i in range(n_experiments):
        print('{} experiment {}'.format(exp_name, i))
        initial_data = data_array[i]
        # build the model-based optimization problem
        opt_problem = BatchProfileOptimizer(
            x_ub, x_lb, g_plant, solver=solver, backoff=backoff)

        # build the adaptation model
        adaptation = MAGaussianProcesses(
            model, initial_data, filter_data=True, neighbors_type=neighbors)

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

        exp_name = 'ma-gp-glw-{}-{}-{}-{}'.format(solver['name'],
                                              neighbors, noise, backoff)

        p = multiprocessing.Process(target=run_rto, args=(
            n_experiments, initial_data_array, solver, db_file, neighbors, exp_name, noise, backoff))
        p.start()
        jobs.append(p)

    # wait for all to finish
    [job.join() for job in jobs]

config = [
        #   {'solver': {'name': 'slsqp_scipy'},
        #    'db_file': '/mnt/d/rto_data/rto_thesis_experiments.db',
        #    'neighbors': 'k_last',
        #    'noise': 0.01,
        #    'backoff': 0.0},
          {'solver': {'name': 'de_scipy_best1bin'},
           'db_file': '/mnt/d/rto_data/rto_thesis_experiments.db',
           'neighbors': 'k_last',
           'noise': 0.01,
           'backoff': 0.00},
           ]


run_rto_experiment(n_experiments, data_size, initial_data_noise, config)

