

import multiprocessing
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from optimization.utils import build_F, calculate_SSE, convert_ivp_results
from model.semi_batch import SemiBatchReactor
from optimization.model_optimization import ProfileOptimizer, ModelParameterOptimizer
from optimization.de import DifferentialEvolution
from bussineslogic.rto_data import RTODataModel

def filter_parameters(calibrated, previous, theta):
    return previous * (1 - theta) + theta * calibrated

# Experiment design
# RTO cycle: 
#   - 10 batches
#   - additive random noise 
#   - new samples generated in each cycle
#   - 30 executions for each cycle
#
# KPIs
# - dF_iteration = 100 * (F(k) - F(k-1))/F(k-1)
# - dL_iteration = 100 * (L(k) - L(k-1))/(L(k-1))
# - dL_opt = 100 * (L - Lopt)/(L_opt)
# - dF_opt = 100 * ||(F - Fopt)/Fopt||
# - parameters distribution
# - contraints violations
# 
# Optimizers
# 1) DE mean/1/bin pop 20, gen 100
# 2) DE rand/1/bin pop 20, gen 100

def run_rto(de_type, rto_runs, rto_cycle):
    theta = 0.5
    pop_size = 20
    max_gen = 100
    sample_times = [1.0] #[0.95, 0.97, 0.99]
    initial_parameters = [0.053, 0.128]

    # Get an instance of the profile and parameter optimizers
    opt = ProfileOptimizer()
    cal = ModelParameterOptimizer()

    # Load the real model to generate samples
    model_ideal = SemiBatchReactor()

    for r in range(rto_runs):
        # Creates the instance in the DB
        md = RTODataModel()
        rto_id = md.create_rto('de:{}, pop:{}, max_gen:{}, run: {}'.format(de_type, pop_size, max_gen, r), rto_type='two-step-paper')

        calibrated_parameters = initial_parameters

        # Begin the RTO loop
        for i in range(0, rto_cycle):
            # First, optimize the input signal using the previous parameters
            # The parameters are filtered, based on chachuat2009
            k1, k2 = filter_parameters(np.asarray(calibrated_parameters), np.asarray(initial_parameters), theta)
            model_aprox = SemiBatchReactor(k=[k1, k2, 0, 0, 5])
            f_cost, f_input, _, _, _ = opt.run(model_aprox, max_gen, pop_size, de_type)
            print('Fopt---> {}'.format(f_cost))
            print('Xopt---> {}'.format(f_input))

            # Then, generate the samples from ideal model, i.e, use the input signal on the plant
            samples = model_ideal.get_samples(f_input, sample_times)

            # And finally calibrate the model parameters
            initial_parameters = calibrated_parameters
            calibration_error, calibrated_parameters, _, _ = cal.run(
                model_aprox, f_input, samples, max_gen, pop_size, de_type)
            print('Fcal---> {}'.format(calibration_error))
            print('Xcal---> {}'.format(calibrated_parameters))

            # Store the RTO iteration in the DB
            # Create runs
            run_id = md.create_run(rto_id, i, 'completed')
            # Samples
            md.save_samples(run_id, samples)
            # Parameters
            md.save_parameters(run_id, {'k1_initial': k1, 'k2_initial': k2,
                                        'k1_calibrated': calibrated_parameters[0], 
                                        'k2_calibrated': calibrated_parameters[1]})
            # Input data
            input_dict = {'tm': f_input[0], 'Fs': f_input[1],'ts': f_input[2]}
            md.save_input_data(run_id, input_dict)

            # Simulation results
            sim_ideal = convert_ivp_results(model_ideal.simulate(f_input), [
                                            'Ca', 'Cb', 'Cc', 'Cd', 'V'])
            sim_initial = convert_ivp_results(model_aprox.simulate(f_input), [
                                            'Ca', 'Cb', 'Cc', 'Cd', 'V'])
            model_calibrated = SemiBatchReactor(
                k=[calibrated_parameters[0], calibrated_parameters[1], 0, 0, 5])
            sim_calibrated = convert_ivp_results(model_calibrated.simulate(f_input), [
                                                'Ca', 'Cb', 'Cc', 'Cd', 'V'])
            # Already save the input signal in a time base
            time = model_ideal.simulate(f_input).t
            timebased_f_dict = {'F': np.transpose(
                np.vstack((time, build_F(time, f_input))))}

            md.save_simulation_results(run_id, sim_initial, 'initial')
            md.save_simulation_results(run_id, sim_ideal, 'ideal')
            md.save_simulation_results(run_id, sim_calibrated, 'calibrated')
            md.save_simulation_results(run_id, timebased_f_dict, 'input')

            # Results
            results_dict = {'error_calibrator': calibration_error, 'cost_optmizer': -f_cost,
                            'Cb_tf_real': sim_ideal['Cb'][-1, 1], 'Cd_tf_real': sim_ideal['Cd'][-1, 1],
                            'Cb_tf_calibr': sim_calibrated['Cb'][-1, 1], 'Cd_tf_calibr': sim_calibrated['Cd'][-1, 1],
                            'Cb_tf_init': sim_initial['Cb'][-1, 1], 'Cd_tf_init': sim_initial['Cd'][-1, 1],
                            'cost_real': sim_ideal['Cc'][-1, 1] * sim_ideal['V'][-1, 1], 
                            'cost_calibr': sim_calibrated['Cc'][-1, 1] * sim_calibrated['V'][-1, 1], 
                            'cost_initial': sim_initial['Cc'][-1, 1] * sim_initial['V'][-1, 1], 
                            'k1_initial': k1, 'k2_initial': k2,
                            'k1_calibrated': calibrated_parameters[0], 'k2_calibrated': calibrated_parameters[1],
                            'tm': f_input[0], 'Fs': f_input[1],'ts': f_input[2]}
            md.save_results(run_id, results_dict)
            
            print('Finished RTO cycle #{}'.format(i))
        print('Finished RTO run #{}'.format(r))


   

if __name__ == '__main__':
    # Define general constants
    rto_runs = 30
    rto_cycle = 10
    de_types = ['mean/1/bin', 'rand/1/bin']
    jobs = []
    for de_type in de_types:
        p = multiprocessing.Process(target=run_rto, args=(de_type, rto_runs, rto_cycle))
        jobs.append(p)
        p.start()
    [job.join() for job in jobs]