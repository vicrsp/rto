from bussineslogic.rto_data import RTODataModel
import os
import sys
import numpy as np

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)


class RTO:
    def __init__(self, process_model, real_process, optimization_problem, adaptation_strategy, cycles=10, db_file=r"\mnt\d\rto\src\data\rto_test.db"):
        # self.md = RTODataModel(db_file)
        self.cycles = cycles
        self.optimization_problem = optimization_problem
        self.adaptation_strategy = adaptation_strategy
        self.process_model = process_model
        self.real_process = real_process

    def set_num_cycles(self, cycles):
        self.cycles = cycles

    def run(self):
        for _ in range(self.cycles):
            f_cost, f_input = self.optimization_problem.run(
                self.process_model, self.adaptation_strategy)

            sim_real = self.real_process.simulate(f_input)
            sim_model = self.process_model.simulate(f_input)

            fr, gr = self.real_process.get_objective(
                sim_real), self.real_process.get_constraints(sim_real)

            gm = self.process_model.get_constraints(sim_model)
            data = np.append(np.asarray(fr - f_cost), gr - gm)

            self.adaptation_strategy.adapt(f_input, data)
