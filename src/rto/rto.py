import logging
import numpy as np
from timeit import default_timer as timer
from .experiment.results_handler import ExperimentResultsHandler
from scipy.stats import norm

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

    def _filter_input(self, xnew, xold):
        return xold + (xnew - xold) * self.k_filter

    def _calculate_results(self, f_input):
        fr, gr = self.real_process.get_objective(
            f_input, self.noise_level), self.real_process.get_constraints(f_input, self.noise_level)

        fm, gm = self.process_model.get_objective(
            f_input), self.process_model.get_constraints(f_input)

        return fr, gr, fm, gm

    def run(self, u_0):
        u_current = u_0
        u_previous = u_0
        rto_id = self._initialize()
        initial_data_size = len(self.adaptation_strategy.initial_data.u)

        for i in range(self.iterations):
            iteration = i + initial_data_size
            # Solve the model-based optimization problem
            u_current, opt_time, n_fev = self._solve_optimization_problem(u_current)
            # Define the next operating point
            u_current, opt_feasible = self._define_next_operating_point(u_current, u_previous)
            # Calculate the results
            fr, gr, fm, gm = self._calculate_results(u_current)
            # Save the results
            run_id = self.experiment.save_results(rto_id, iteration, fr, gr, fm, gm, u_current, opt_feasible, opt_time, n_fev)
            # Save the trained models
            self._save_models(run_id)
            # Get GP training data
            data = self._get_training_data(fr, fm, gr, gm)
            # Execute the adaptation strategy
            self.adaptation_strategy.adapt(u_current, data)
            # update the current solution
            u_previous = u_current
           
        return rto_id

    def _get_training_data(self, fr, fm, gr, gm):
        data = np.append(np.asarray(fr - fm), gr - gm)
        return data

    def _save_models(self, run_id):
        ma_models = {}
        ma_models['gp_scaler'] = self.adaptation_strategy.input_scaler
        ma_models['f'] = self.adaptation_strategy.models[0]
        [ma_models.update({f'g_{i}': model }) for i, model in enumerate(self.adaptation_strategy.models[1:])] 
        self.experiment.save_models(run_id, ma_models)
        return ma_models

    def _define_next_operating_point(self, u_current, u_previous):
        opt_feasible = True
        if(len(u_current) > 0):
            u_current = self._filter_input(u_current, u_previous)
        else:
            u_current = u_previous * (1 + np.random.normal(scale=0.01))
            # ensure new random point is inside feasible region
            u_current = np.maximum(u_current, self.optimization_problem.lb)
            u_current = np.minimum(u_current, self.optimization_problem.ub)
            opt_feasible = False

            logging.warning('unfeasible optimization result. using random generated point.')
        return u_current, opt_feasible

    def _solve_optimization_problem(self, u_current):
        start = timer()
        _, u_current, n_fev = self.optimization_problem.run(self.process_model, u_current, adaptation_strategy=self.adaptation_strategy)
        end = timer()
        opt_time = end - start

        return u_current, opt_time, n_fev

    def _initialize(self):
        rto_id = self.experiment.create_rto()        
        self.experiment.save_initial_data(self.adaptation_strategy.initial_data, rto_id)
        return rto_id


class RTOBayesian(RTO):
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, iterations, name='bayesian', db_file='/mnt/d/rto_data/rto_test.db', noise=0.01):
        super().__init__(process_model, real_process, optimization_problem, adaptation_strategy, iterations, db_file, name, noise)
        self._delta_c = 0.01

    def run(self, u_0):
        u_current = u_0
        u_previous = u_0
        rto_id = self._initialize()
        best_plant_objective = self._get_best_initial_solution()
        initial_data_size = len(self.adaptation_strategy.initial_data.u)

        for i in range(self.iterations):
            iteration = i + initial_data_size
            # Solve the model-based optimization problem
            u_current, opt_time, n_fev = self._solve_optimization_problem(u_current, best_plant_objective)
            # Define the next operating point
            u_current, opt_feasible = self._define_next_operating_point(u_current, u_previous)
            # Calculate the results
            fr, gr, fm, gm = self._calculate_results(u_current)
            # Get GP training data
            data = self._get_training_data(fr, fm, gr, gm)
            # Save the results
            run_id = self.experiment.save_results(rto_id, iteration, fr, gr, fm, gm, u_current, opt_feasible, opt_time, n_fev)
            # save the models
            self._save_models(run_id)
            # save the best plant solution
            self.save_best_plant_solution(best_plant_objective, run_id)
            # Update the best solution
            best_plant_objective = self._update_best_solution(best_plant_objective, fr, gr)
            # Execute the adaptation strategy
            self.adaptation_strategy.adapt(u_current, data)
            # update the current solution
            u_previous = u_current
           
        return rto_id

    def save_best_plant_solution(self, best_plant_objective, run_id):
        self.experiment.save_run_results(run_id, {'best_plant_objective': best_plant_objective})

    def _get_best_initial_solution(self):
        measurements = self.adaptation_strategy.initial_data.measurements
        g = self.optimization_problem.g
        fr, gr = measurements[:, 0], measurements[:, 1]
        # find feasible points
        feasible_points = [np.all(gri <= g) for gri in gr]
        if np.all(feasible_points == False): 
            logging.warning('Initial points are all unfeasible. The optimizer will be guided by feasibility only.')
            return None
        # then the best one among them
        best_plant_objective = np.min(fr[feasible_points])
        return best_plant_objective

    def _update_best_solution_known_variances(self, current_best, fr, gr, variances):
        var_f, var_g = variances
        f_prob = norm.cdf((current_best - fr)/var_f)
        g_probs = np.array([norm.cdf((self.optimization_problem.g[i] - gr[i])/var_gi) for i, var_gi in enumerate(var_g)])

        if((f_prob >= 1 - self._delta_c) & np.all(g_probs >= 1 - self._delta_c)):
            return fr
        return current_best

    
    def _update_best_solution_unknown_variances(self, current_best, fr, gr):
        if((fr < current_best) & np.all(gr <= self.optimization_problem.g)):
            return fr
        return current_best

    def _update_best_solution(self, current_best, fr, gr, variances=None):
        # should we calculate the probability of the current solution being better than the best?
        # YES, but it would be only possible if we KNEW the process noise variance !!!.
        if(variances is None): 
            return self._update_best_solution_unknown_variances(current_best, fr, gr)
        # When we have it, lets use!
        return self._update_best_solution_known_variances(current_best, fr, gr, variances)
    
    def _solve_optimization_problem(self, u_current, best_plant_objective):
        start = timer()
        _, u_current, n_fev = self.optimization_problem.run(self.process_model, u_current, adaptation_strategy=self.adaptation_strategy, f_best=best_plant_objective)
        end = timer()
        opt_time = end - start

        return u_current, opt_time, n_fev