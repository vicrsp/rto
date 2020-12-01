from bussineslogic.rto_data import RTODataModel
import os
import sys
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)


class RTO:
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, iterations=10, db_file='/mnt/d/rto_data/rto_test.db', name='ma-gp', noise=0.01):
        self.md = RTODataModel(db_file)
        self.iterations = iterations
        self.optimization_problem = optimization_problem
        self.adaptation_strategy = adaptation_strategy
        self.process_model = process_model
        self.real_process = real_process
        self.k_filter = 0.4
        self.noise_level = noise  # %
        self.experiment_name = name

    def set_iterations(self, iterations):
        self.iterations = iterations

    def filter_input(self, xnew, xold):
        return xold + (xnew - xold) * self.k_filter

    def save_initial_data(self, initial_data, rto_id):
        for index, u0_i in enumerate(initial_data[0]):
            _, fr, gr, fm, gm = self.calculate_results(u0_i)
            self.save_results(rto_id, index, fr, gr, fm,
                              gm, u0_i, True, 0, 'initialization')

    def save_results(self, rto_id, index, fr, gr, fm, gm, f_input, opt_feasible, opt_time, run_type='closed-loop'):
        run_id = self.md.create_run(rto_id, index, run_type)
        results_dict = {'cost_real': fr, 'cost_model': fm,
                        'fobj_modifier': fr - fm, 'g_modifiers': ','.join(str(v) for v in (gr-gm)),
                        'g_real': ','.join(str(v) for v in gr), 'g_model': ','.join(str(v) for v in gm),
                        'u': ','.join(str(v) for v in f_input),
                        'opt_feasible': str(opt_feasible), 'opt_time': opt_time}
        self.md.save_results(run_id, results_dict)
        return run_id

    def calculate_results(self, f_input):
        sim_real = self.real_process.simulate(f_input)
        sim_model = self.process_model.simulate(f_input)
        # with 1% gaussian noise
        fr, gr = self.real_process.get_objective(
            sim_real, self.noise_level), self.real_process.get_constraints(f_input, sim_real, self.noise_level)

        fm, gm = self.process_model.get_objective(
            sim_model), self.process_model.get_constraints(f_input, sim_model)
        data = np.append(np.asarray(fr - fm), gr - gm)

        return data, fr, gr, fm, gm

    def run(self, u_0):
        rto_id = self.md.create_rto(
            'test at {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), rto_type=self.experiment_name)
        f_input = u_0
        f_previous = u_0

        self.save_initial_data(self.adaptation_strategy.initial_data, rto_id)
        initial_data_size = len(self.adaptation_strategy.initial_data[0])

        for i in range(self.iterations):
            iteration = i + initial_data_size
            print('{}: iteration {} started!'.format(
                self.experiment_name, iteration))

            start = timer()
            _, f_input = self.optimization_problem.run(
                self.process_model, self.adaptation_strategy, f_input)
            end = timer()
            opt_time = end - start

            opt_feasible = True
            if(len(f_input) > 0):
                f_input = self.filter_input(f_input, f_previous)
            else:
                f_input = f_previous * (1 + np.random.normal(scale=0.01))
                # ensure new random point is feasible
                f_input = np.maximum(f_input, self.optimization_problem.lb)
                f_input = np.minimum(f_input, self.optimization_problem.ub)
                opt_feasible = False

                print('unfeasible optimization result. using random generated point.')

            f_previous = f_input

            # Calculate the results
            data, fr, gr, fm, gm = self.calculate_results(f_input)
            # Exexute the adaptation strategy
            self.adaptation_strategy.adapt(f_input, data)
            # Save the results
            self.save_results(rto_id, iteration, fr, gr, fm,
                              gm, f_input, opt_feasible, opt_time)
            print('{}: cost_model= {}; cost_real= {}'.format(
                self.experiment_name, fm, fr))
