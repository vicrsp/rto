from bussineslogic.rto_data import RTODataModel
import os
import sys
import numpy as np
from datetime import datetime

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)


class RTO:
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, cycles=10, db_file='/mnt/d/rto_data/rto_test.db'):
        self.md = RTODataModel(db_file)
        self.cycles = cycles
        self.optimization_problem = optimization_problem
        self.adaptation_strategy = adaptation_strategy
        self.process_model = process_model
        self.real_process = real_process

    def set_num_cycles(self, cycles):
        self.cycles = cycles

    def run(self):

        rto_id = self.md.create_rto(
            'test at {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), rto_type='ma-gp')

        for i in range(self.cycles):
            print('Cycle {} started!'.format(i))
            f_cost, f_input = self.optimization_problem.run(
                self.process_model, self.adaptation_strategy)

            sim_real = self.real_process.simulate(f_input)
            sim_model = self.process_model.simulate(f_input)

            fr, gr = self.real_process.get_objective(
                sim_real), self.real_process.get_constraints(f_input, sim_real)

            gm = self.process_model.get_constraints(f_input, sim_model)
            data = np.append(np.asarray(fr - f_cost), gr - gm)

            model_scores = self.adaptation_strategy.adapt(f_input, data)

            run_id = self.md.create_run(rto_id, i, 'completed')
            results_dict = {'cost_real': fr, 'cost_model': f_cost,
                            'fobj_modifier': fr - f_cost, 'g_modifiers': ','.join(str(v) for v in (gr-gm)),
                            'g_real': ','.join(str(v) for v in gr), 'g_model': ','.join(str(v) for v in gm),
                            # 'Cb_tf_real': sim_real['Cb'][-1, 1], 'Cd_tf_real': sim_real['Cd'][-1, 1],
                            # 'Cc_tf_real': sim_real['Cc'][-1, 1],
                            'u': ','.join(str(v) for v in f_input),
                            'gp_scores': ','.join(str(v) for v in model_scores)}
            self.md.save_results(run_id, results_dict)
