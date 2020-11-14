from bussineslogic.rto_data import RTODataModel
import os
import sys
import numpy as np
from datetime import datetime

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)


class RTO:
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, iterations=10, db_file='/mnt/d/rto_data/rto_test.db', name='ma-gp'):
        self.md = RTODataModel(db_file)
        self.iterations = iterations
        self.optimization_problem = optimization_problem
        self.adaptation_strategy = adaptation_strategy
        self.process_model = process_model
        self.real_process = real_process
        self.k_filter = 0.4
        self.delta_input = 0.3  # %
        self.noise_level = 0.01  # %
        self.experiment_name = name

    def set_iterations(self, iterations):
        self.iterations = iterations

    def filter_input(self, xnew, xold):
        return xold + (xnew - xold) * self.k_filter

    def run(self, u_0):

        rto_id = self.md.create_rto(
            'test at {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), rto_type=self.experiment_name)
        f_input = u_0
        for i in range(self.iterations):
            print('{}: iteration {} started!'.format(self.experiment_name, i))

            # (1 - self.delta_input) * np.asarray(f_input)
            lower_bound = self.optimization_problem.lb
            # (1 + self.delta_input) * np.asarray(f_input)
            upper_bound = self.optimization_problem.ub

            #lower_bound = np.maximum(lower_bound, self.optimization_problem.lb)
            #upper_bound = np.minimum(upper_bound, self.optimization_problem.ub)

            _, f_input = self.optimization_problem.run(
                self.process_model, self.adaptation_strategy, lower_bound, upper_bound, f_input)

            if(i > 0):
                f_input = self.filter_input(f_input, f_previous)
            f_previous = f_input

            sim_real = self.real_process.simulate(f_input)
            sim_model = self.process_model.simulate(f_input)
            # with 5% gaussian noise
            fr, gr = self.real_process.get_objective(
                sim_real, self.noise_level), self.real_process.get_constraints(f_input, sim_real, self.noise_level)

            fm, gm = self.process_model.get_objective(
                sim_model), self.process_model.get_constraints(f_input, sim_model)
            data = np.append(np.asarray(fr - fm), gr - gm)

            self.adaptation_strategy.adapt(f_input, data)

            run_id = self.md.create_run(rto_id, i, 'completed')
            results_dict = {'cost_real': fr, 'cost_model': fm,
                            'fobj_modifier': fr - fm, 'g_modifiers': ','.join(str(v) for v in (gr-gm)),
                            'g_real': ','.join(str(v) for v in gr), 'g_model': ','.join(str(v) for v in gm),
                            'u': ','.join(str(v) for v in f_input)}
            self.md.save_results(run_id, results_dict)
            print('{}: u={}'.format(self.experiment_name, fr - fm))
            #print('cost_model: {}'.format(fm))
            #print('cost_real: {}'.format(fr))
            #print('f_input: {}'.format(f_input))
