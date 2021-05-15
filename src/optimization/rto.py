from bussineslogic.rto_data import RTODataModel
import os
import sys
import numpy as np
import pandas as pd
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
        self.results = []

    def set_iterations(self, iterations):
        self.iterations = iterations

    def filter_input(self, xnew, xold):
        return xold + (xnew - xold) * self.k_filter

    def save_initial_data(self, initial_data, rto_id):
        for index, u0_i in enumerate(initial_data[0]):
            fr, gr, fm, gm = initial_data[2][index]
            self.save_results(rto_id, index, fr, gr, fm,
                              gm, u0_i, True, 0, 0, 'initialization')

    def save_results(self, rto_id, index, fr, gr, fm, gm, f_input, opt_feasible, opt_time, n_fev, run_type='closed-loop'):
        run_id = self.md.create_run(rto_id, index, run_type)
        results_dict = {'cost_real': fr, 'cost_model': fm,
                        'fobj_modifier': fr - fm, 'g_modifiers': ','.join(str(v) for v in (gr-gm)),
                        'g_real': ','.join(str(v) for v in gr), 'g_model': ','.join(str(v) for v in gm),
                        'u': ','.join(str(v) for v in f_input),
                        'opt_feasible': str(opt_feasible), 'opt_time': opt_time, 'n_fev': n_fev}
        self.results.append(results_dict)
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
            # print('{}: iteration {} started!'.format(
            #     self.experiment_name, iteration))

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

                print('unfeasible optimization result. using random generated point.')

            f_previous = f_input

            # Calculate the results
            data, fr, gr, fm, gm = self.calculate_results(f_input)
            # Exexute the adaptation strategy
            self.adaptation_strategy.adapt(f_input, data)
            # Save the results
            self.save_results(rto_id, iteration, fr, gr, fm,
                              gm, f_input, opt_feasible, opt_time, n_fev)
            print('[{}]-[{}]: iteration={}'.format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), self.experiment_name, iteration))
        return rto_id

    def pre_process_results(self, all_results, f_plant):
        def aggfunc(x):
            return x

        # Transform the data
        all_results_pv = pd.pivot_table(all_results, values='value', index=['run.id','iteration','rto.type'], columns=['var_name'], aggfunc=aggfunc)
        all_results_pv.reset_index(level=all_results_pv.index.names, inplace=True)
        
        # remove the suffix
        all_results_pv['rto.type'] = all_results_pv['rto.type'].apply(lambda x: x.split('-')[2])

        # Convert the values
        all_results_pv[['cost_model','cost_real','fobj_modifier', 'opt_time']] = all_results_pv[['cost_model','cost_real','fobj_modifier','opt_time']].astype('float')

        # Extract some variables
        # all_results_pv['g_0'] = all_results_pv['g_real'].apply(lambda x: float(x.split(',')[0])) 
        # all_results_pv['g_1'] = all_results_pv['g_real'].apply(lambda x: float(x.split(',')[1])) 
        # all_results_pv['g_0_model'] = all_results_pv['g_model'].apply(lambda x: float(x.split(',')[0])) 
        # all_results_pv['g_1_model'] = all_results_pv['g_model'].apply(lambda x: float(x.split(',')[1])) 
        # all_results_pv['g_0_modifiers'] = all_results_pv['g_modifiers'].apply(lambda x: float(x.split(',')[0])) 
        # all_results_pv['g_1_modifiers'] = all_results_pv['g_modifiers'].apply(lambda x: float(x.split(',')[1])) 

        # all_results_pv['tm'] = all_results_pv['u'].apply(lambda x: float(x.split(',')[0])) 
        # all_results_pv['Fs'] = all_results_pv['u'].apply(lambda x: float(x.split(',')[1])) 
        # all_results_pv['ts'] = all_results_pv['u'].apply(lambda x: float(x.split(',')[2])) 

        # kpis
        # all_results_pv['du'] = all_results_pv[['tm','Fs','ts']].apply(lambda x: np.linalg.norm(100 * (x - u_plant)/u_plant), axis=1)
        all_results_pv['dPhi'] = all_results_pv[['cost_real']].apply(lambda x: 100 * np.abs((x - f_plant)/f_plant))
        # all_results_pv['g_Cb_tf'] = all_results_pv['g_0'].apply(lambda x: 'Not violated' if x <= 0.025 else 'Violated')
        # all_results_pv['g_Cd_tf'] = all_results_pv['g_1'].apply(lambda x: 'Not violated' if x <= 0.15 else 'Violated')

        return all_results_pv

    def calculate_performance(self, rto_id, rto_type, f_plant):
        data = pd.DataFrame(self.md.get_rto_results(rto_id, rto_type), 
                        columns=['rto.id', 'rto.name', 'rto.type', 'run.id', 'iteration', 'var_name', 'value'])
        data_pp = self.pre_process_results(data, f_plant)
        #return float(data_pp['dPhi'].iloc[-1]) # final gap
        return np.trapz(data_pp['dPhi']) # AUC