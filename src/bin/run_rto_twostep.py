

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

def filtered_parameters(calibrated, previous, theta):
    return previous * (1 - theta) + theta * calibrated


# Define general constants
rto_runs = 20
n_samples = 3

# Creates the instance in the DB
md = RTODataModel()
rto_id = md.create_rto('theta=1')

# Load the real model to generate samples
model_ideal = SemiBatchReactor()

# Get an instance of the profile and parameter optimizers
opt = ProfileOptimizer()
cal = ModelParameterOptimizer()

initial_parameters = [0.053, 0.128]
calibrated_parameters = initial_parameters

# Begin the RTO loop
for i in range(0, rto_runs):
    # First, optimize the input signal using the previous parameters
    # The parameters are filtered, based on chachua2009
    k1, k2 = filtered_parameters(np.asarray(calibrated_parameters), np.asarray(initial_parameters), 1.0)
    model_aprox = SemiBatchReactor(k=[k1, k2, 0, 0, 5])
    f_cost, f_input, _, _, _ = opt.run(model_aprox)
    print('Fopt---> {}'.format(f_cost))
    print('Xopt---> {}'.format(f_input))

    # Then, generate the samples from ideal model, i.e, use the input signal on the plant
    samples = model_ideal.get_samples(f_input, [0.95, 0.97, 0.99])

    # And finally calibrate the model parameters
    initial_parameters = calibrated_parameters
    calibration_error, calibrated_parameters, _, _ = cal.run(
        model_aprox, f_input, samples)
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
    input_dict = {'F0': f_input[0], 'tm': f_input[1],
                  'Fm': f_input[2], 'ts': f_input[3], 'Fs': f_input[4]}
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
    results_dict = {'error': calibration_error, 'cost_model': -f_cost,
                    'Cb_tf': sim_ideal['Cb'][-1, 1], 'Cd_tf': sim_ideal['Cd'][-1, 1],
                    'cost_real': sim_ideal['Cc'][-1, 1] * sim_ideal['V'][-1, 1], 
                    'k1_initial': k1, 'k2_initial': k2,
                    'k1_calibrated': calibrated_parameters[0], 'k2_calibrated': calibrated_parameters[1],
                    'F0': f_input[0], 'tm': f_input[1],
                    'Fm': f_input[2], 'ts': f_input[3], 'Fs': f_input[4]}
    md.save_results(run_id, results_dict)