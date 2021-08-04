import logging
import numpy as np
from timeit import default_timer as timer
from .experiment.results_handler import ExperimentResultsHandler

logging.basicConfig(format='[%(asctime)s]:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

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
        u_current = u_0
        u_previous = u_0
        rto_id, best_plant_objective = self.initialize_rto_cycle()
        initial_data_size = len(self.adaptation_strategy.initial_data.u)

        for i in range(self.iterations):
            iteration = i + initial_data_size
            # Solve the model-based optimization problem
            u_current, opt_time, n_fev = self.solve_model_based_optimization_problem(u_current, best_plant_objective)
            # Define the next operating point
            u_current, opt_feasible = self.define_next_operating_point(u_current, u_previous)
            # Calculate the results
            data, fr, gr, fm, gm = self.calculate_results(u_current)
            # Update the best solution
            if((fr < best_plant_objective) & np.all(gr <= self.optimization_problem.g)):
                best_plant_objective = fr
            # Updates the operating point based on the adaptation strategy rule
            u_updated = self.adaptation_strategy.update_operating_point(u_current, data)
            # Execute the adaptation strategy
            self.adaptation_strategy.adapt(u_updated, data)
            # Save the results
            self.experiment.save_results(rto_id, iteration, fr, gr, fm, gm, u_updated, opt_feasible, opt_time, n_fev)
            # update the current solution
            u_previous = u_updated
           
        return rto_id

    def define_next_operating_point(self, u_current, u_previous):
        opt_feasible = True
        if(len(u_current) > 0):
            u_current = self.filter_input(u_current, u_previous)
        else:
            u_current = u_previous * (1 + np.random.normal(scale=0.01))
            # ensure new random point is inside feasible region
            u_current = np.maximum(u_current, self.optimization_problem.lb)
            u_current = np.minimum(u_current, self.optimization_problem.ub)
            opt_feasible = False

            logging.warning('unfeasible optimization result. using random generated point.')
        return u_current, opt_feasible

    def solve_model_based_optimization_problem(self, u_current, best_plant_objective):
        start = timer()
        _, u_current, n_fev = self.optimization_problem.run(
            self.process_model, self.adaptation_strategy, u_current, best_plant_objective)
        end = timer()
        opt_time = end - start

        return u_current, opt_time, n_fev

    def initialize_rto_cycle(self):
        rto_id = self.experiment.create_rto()        
        self.experiment.save_initial_data(self.adaptation_strategy.initial_data, rto_id)
        best_solution = np.min(self.adaptation_strategy.initial_data.measurements[:,0])
        return rto_id, best_solution
