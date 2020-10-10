

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

# Define general constants
rto_runs = 10
n_samples = 3
initial_parameters = [0.053, 0.128]

# Creates the instance in the DB
md = RTODataModel()
rto_id = md.create_rto()

# Define variables to store the progress
run_samples = []
run_simulated_samples = []
run_parameters = []
run_inputs = []
run_calibration_objective = []
run_optimization_objective = []

# Load the real model to generate samples
model_ideal = SemiBatchReactor()

# Get an instance of the profile and parameter optimizers
opt = ProfileOptimizer()
cal = ModelParameterOptimizer()

calibrated_parameters = initial_parameters

# Begin the RTO loop
for i in range(0, rto_runs):
    # First, optimize the input signal using the previous parameters
    k1, k2 = calibrated_parameters
    model_aprox = SemiBatchReactor(k=[k1, k2, 0, 0, 5])
    f_cost, f_input, _, _, _ = opt.run(model_aprox)
    print('Fopt---> {}'.format(f_cost))
    print('Xopt---> {}'.format(f_input))

    # Then, generate the samples from ideal model, i.e, use the input signal on the plant
    samples = model_ideal.get_samples(f_input, [0.95, 0.97, 0.99])

    # And finally calibrate the model parameters
    calibration_error, calibrated_parameters, _, _ = cal.run(
        model_aprox, f_input, samples)
    print('Fcal---> {}'.format(calibration_error))
    print('Xcal---> {}'.format(calibrated_parameters))

    # Store the RTO iteration in the DB
    # Create runs
    ro_id = md.create_run(rto_id, 'optimization', i, 'completed')
    rc_id = md.create_run(rto_id, 'calibration', i, 'completed')
    # Samples
    md.save_samples(rc_id, samples)
    # Parameters
    md.save_parameters(ro_id, {'k1': k1, 'k2': k2})
    md.save_parameters(
        rc_id, {'k1': calibrated_parameters[0], 'k2': calibrated_parameters[1]})
    # Input data
    input_dict = {'F0': f_input[0], 'tm': f_input[1],
                  'Fm': f_input[2], 'ts': f_input[3], 'Fs': f_input[4]}
    md.save_input_data(ro_id, input_dict)
    md.save_input_data(rc_id, input_dict)
    # Results
    rc_results_dict = {'error': calibration_error}
    ro_results_dict = {'cost_function': -f_cost}
    md.save_results(rc_id, rc_results_dict)
    md.save_results(ro_id, ro_results_dict)

    # Simulation results    
    sim_ideal = convert_ivp_results(model_ideal.simulate(f_input), ['Ca','Cb','Cc','Cd','V'])
    sim_initial = convert_ivp_results(model_aprox.simulate(f_input), ['Ca','Cb','Cc','Cd','V'])
    model_calibrated = SemiBatchReactor(
        k=[calibrated_parameters[0], calibrated_parameters[1], 0, 0, 5])
    sim_calibrated = convert_ivp_results(model_calibrated.simulate(f_input), ['Ca','Cb','Cc','Cd','V'])
    # Already save the input signal in a time base
    time = model_ideal.simulate(f_input).t
    timebased_f_dict =  {'F': np.transpose(np.vstack((time, build_F(time, f_input))))}

    # Non-calibrated + ideal
    md.save_simulation_results(ro_id, sim_initial, 'estimated')
    md.save_simulation_results(ro_id, sim_ideal, 'expected')
    # Calibrated + ideal
    md.save_simulation_results(rc_id, sim_calibrated, 'estimated')
    md.save_simulation_results(rc_id, sim_ideal, 'expected')
    # Time-based input
    md.save_simulation_results(ro_id, timebased_f_dict, 'input')
    md.save_simulation_results(rc_id, timebased_f_dict, 'input')
