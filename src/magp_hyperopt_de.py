import numpy as np
import multiprocessing
import GPyOpt
from datetime import datetime

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform

n_experiments = 1
n_iterations = 20
data_size = 5
initial_data_noise = 0.01

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.4427644, 0.00110823777, 227.792418]
f_real_optimum = -0.5085930760109818
u_0 = [15.652103265931729, 0.0007141834799295323, 225.48063936756103]
g_plant = np.array([0.025, 0.15])
x_ub = [30, 0.002, 250]
x_lb = [0, 0, 200]

strategy_map = ['best1bin', 'rand1bin', 'randtobest1bin', 'best2bin', 'rand2bin', 'best1exp', 'rand1exp']

class RTOHyperOpt:
    def __init__(self, n_experiments, data_array, solver, db_file, neighbors, exp_name, noise, backoff):
        self.n_experiments = n_experiments
        self.data_array = data_array
        self.solver = solver
        self.db_file = db_file
        self.neighbors = neighbors
        self.exp_name = exp_name
        self.noise = noise
        self.backoff = backoff

    def cost_function(self, x):
        # build the model-based optimization problem
        def parse_x(x):
            params = {}
            for index, domain_cfg in enumerate(self.solver['domain']):
                value = x.ravel()[index]
                if(domain_cfg['type'] == 'discrete'):
                    params[domain_cfg['name']] = int(value)
                else:
                    params[domain_cfg['name']] = value

            params['strategy'] = strategy_map[int(params['strategy'])]
            return params

        de_params = parse_x(x) 
        eval_name = self.exp_name
        for k,v in de_params.items():
            eval_name = f'{eval_name}-{k}:{v}'
        
        iteration_perf = []
        
        for i in range(self.n_experiments):
            print('[{}]-[{}]: experiment={}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), eval_name, i))
            initial_data = self.data_array[i]

            solver_params={'name': self.solver['name'], 'params': de_params }
            opt_problem = BatchProfileOptimizer(
                x_ub, x_lb, g_plant, solver=solver_params, backoff=self.backoff)

            # build the adaptation model
            adaptation = MAGaussianProcesses(
                model, initial_data, filter_data=True, neighbors_type=self.neighbors)

            u_0_feas = initial_data[0][-1]
            rto = RTO(model, plant, opt_problem, adaptation,
                    iterations=n_iterations, db_file=self.db_file, name=eval_name, noise=self.noise)

            rto_id = rto.run(u_0_feas)
            perf = rto.calculate_performance(rto_id, eval_name, f_real_optimum)     
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]-[{eval_name}]: PERF={perf}')
            iteration_perf.append(perf)

        auc = np.mean(iteration_perf)
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]-[{eval_name}]: Avg. PERF={auc}')
        return auc

def run_rto_experiment(n_experiments, initial_data_size, initial_data_noise, config, max_threads=4):

    # generate all the data before
    initial_data_array = [generate_samples_uniform(
        model, plant, g_plant, u_0, initial_data_size, noise=initial_data_noise) for i in range(n_experiments)]

    for cfg in config:
        solver = cfg['solver']
        code = cfg['code']
        db_file = cfg['db_file']
        neighbors = cfg['neighbors']
        noise = cfg['noise']
        backoff = cfg['backoff']

        domain = solver['domain']
        exp_name = 'ma-gp-{}-{}-{}-{}'.format(solver['name'], code, neighbors, noise)

        rto_hyopt = RTOHyperOpt(n_experiments, initial_data_array, solver, db_file, neighbors, exp_name, noise, backoff)
        myBopt = GPyOpt.methods.BayesianOptimization(f=rto_hyopt.cost_function,                     # Objective function
                                             domain=domain,          # Box-constraints of the problem
                                             initial_design_numdata = 8,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = False,
                                             batch_size=4,
                                             num_cores=4)       

        max_iter = 50       ## maximum number of iterations
        max_time = 3600 * 12       ## maximum allowed time
        # eps      = 0.001     ## tolerance, max distance between consicutive evaluations.

        myBopt.run_optimization(max_iter,max_time)
        myBopt.plot_convergence(filename='bo_convergence_05.png')
        myBopt.plot_acquisition(filename='bo_acquisition_05.png')
        myBopt.save_report(report_file='bo_report_05.txt')
        myBopt.save_evaluations(evaluations_file='bo_evaluations_05.txt')
        print(myBopt.X)
        print(myBopt.Y)


config = [{
            'code': 'BOP05',
            'solver': {'name': 'de_scipy',
                        'domain': [
                          {'name': 'strategy', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4, 5, 6),'dimensionality': 1},
                          {'name': 'recombination', 'type': 'continuous', 'domain': (0.5, 1.0), 'dimensionality': 1},
                        #   {'name': 'mutation', 'type': 'continuous', 'domain': (0.5, 1.5), 'dimensionality': 1},
                          {'name': 'popsize', 'type': 'discrete', 'domain': (5, 15, 30, 50, 100), 'dimensionality': 1}
                        ]
            },
            'db_file': '/mnt/d/rto_data/rto_poc_de_experiments.db',
            'neighbors': 'k_last',
            'noise': 0.01,
            'backoff': 0.00
        }]


run_rto_experiment(n_experiments, data_size, initial_data_noise, config)
