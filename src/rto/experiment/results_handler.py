from datetime import datetime
import logging
from .data_model import RTODataModel

class ExperimentResultsHandler:
    def __init__(self, name, db_file):
        self.md = RTODataModel(db_file)
        self.name = name

    def create_rto(self):
        rto_id = self.md.create_rto('RTO at {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), rto_type=self.name)
        return rto_id

    def save_initial_data(self, initial_data, rto_id):
        for index, u0_i in enumerate(initial_data.u):
            fr, gr, fm, gm = initial_data.measurements[index]
            self.save_results(rto_id, index, fr, gr, fm,
                              gm, u0_i, True, 0, 0, 'initialization')

    def save_results(self, rto_id, index, fr, gr, fm, gm, f_input, opt_feasible, opt_time, n_fev, run_type='closed-loop'):
        run_id = self.md.create_run(rto_id, index, run_type)
        results_dict = {'cost_real': fr, 'cost_model': fm,
                        'fobj_modifier': fr - fm, 'g_modifiers': ','.join(str(v) for v in (gr-gm)),
                        'g_real': ','.join(str(v) for v in gr), 'g_model': ','.join(str(v) for v in gm),
                        'u': ','.join(str(v) for v in f_input),
                        'opt_feasible': str(opt_feasible), 'opt_time': opt_time, 'n_fev': n_fev}
        self.md.save_results(run_id, results_dict)
        logging.debug(f'[{self.name}]: iteration={index}; cost_real={fr}')

        return run_id