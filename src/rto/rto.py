import logging
import numpy as np
from timeit import default_timer as timer
from .experiment.results_handler import ExperimentResultsHandler

logging.basicConfig(format='[%(asctime)s]:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class RTO:
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, iterations=10, db_file='/mnt/d/rto_data/rto_test.db', name='ma-gp', noise=0.01):
        self.experiment = ExperimentResultsHandler(name, db_file)
        self.iterations = iterations
        self.optimization_problem = optimization_problem
        self.adaptation_strategy = adaptation_strategy
        self.process_model = process_model
        self.real_process = real_process
        self.k_filter = 0.4
        self.noise_level = noise  # %
        self.name = name

    def filter_input(self, xnew, xold):
        return xold + (xnew - xold) * self.k_filter

    def calculate_results(self, f_input):
        fr, gr = self.real_process.get_objective(
            f_input, self.noise_level), self.real_process.get_constraints(f_input, self.noise_level)

        fm, gm = self.process_model.get_objective(
            f_input), self.process_model.get_constraints(f_input)
        data = np.append(np.asarray(fr - fm), gr - gm)

        return data, fr, gr, fm, gm

    def run(self, u_0):
        rto_id = self.experiment.create_rto()
        f_input = u_0
        f_previous = u_0

        self.experiment.save_initial_data(
            self.adaptation_strategy.initial_data, rto_id)
        initial_data_size = len(self.adaptation_strategy.initial_data[0])

        for i in range(self.iterations):
            iteration = i + initial_data_size
            start = timer()
            _, f_input, n_fev = self.optimization_problem.run(
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

                logging.warning('unfeasible optimization result. using random generated point.')

            # Calculate the results
            data, fr, gr, fm, gm = self.calculate_results(f_input)
            # Updates the operating point based on the adaptation strategy rule
            f_input_updated = self.adaptation_strategy.update_operating_point(f_input, data)
             # Execute the adaptation strategy
            self.adaptation_strategy.adapt(f_input_updated, data)
            # Save the results
            self.experiment.save_results(
                rto_id, iteration, fr, gr, fm, gm, f_input, opt_feasible, opt_time, n_fev)
                
            f_previous = f_input_updated
            logging.debug(f'[{self.name}]: iteration={iteration}')
           
        return rto_id
