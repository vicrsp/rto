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

from optimization.utils import build_F, calculate_SSE
from model.semi_batch import SemiBatchReactor
from optimization.model_optimization import ProfileOptimizer, ModelParameterOptimizer
from optimization.de import DifferentialEvolution

# Define general constants
rto_runs = 10
n_samples = 3
initial_parameters = [0.053, 0.128, 0.0, 0.0, 5]

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

# First run
# Get optimal profiles for initial set of params
model_aprox = SemiBatchReactor(k=initial_parameters)
f_aprox, x_aprox, _, _, _ = opt.run(model_aprox)

# Calculate the initial model error
samples = model_ideal.get_samples(x_aprox, [0.95, 0.97, 0.99])
sim_values = model_aprox.get_simulated_samples(
    x_aprox, [initial_parameters[0], initial_parameters[1]], samples)
error_approx = calculate_SSE(sim_values, samples)

# Store the initial vales
run_inputs.append(x_aprox)
run_optimization_objective.append(f_aprox)
run_parameters.append(initial_parameters)
run_samples.append(samples)
run_simulated_samples.append(sim_values)
run_calibration_objective.append(error_approx)

input_signal = x_aprox

# Begin the RTO loop
for i in range(1, rto_runs):
    # First step: calibrate the model parameters
    calibration_error, calibrated_parameters, _, _ = cal.run(
        model_aprox, input_signal, samples)
    print('Fcal---> {}'.format(calibration_error))
    print('Xcal---> {}'.format(calibrated_parameters))

    # Second step: Optimize the input signal using the previous parameters
    k1, k2 = calibrated_parameters
    model_adjusted = SemiBatchReactor(k=[k1, k2, 0, 0, 5])
    f_cost, f_input, _, _, _ = opt.run(model_adjusted)
    print('Fopt---> {}'.format(f_cost))
    print('Xopt---> {}'.format(f_input))

    # Store the RTO iteration values
    run_inputs.append(f_input)
    run_optimization_objective.append(f_cost)
    run_parameters.append([k1, k2, 0, 0, 5])
    run_samples.append(samples)
    run_simulated_samples.append(model_aprox.get_simulated_samples(
        input_signal, calibrated_parameters, samples))
    run_calibration_objective.append(calibration_error)

    # Update values for next iteration
    samples = model_ideal.get_samples(f_input, [0.95, 0.97, 0.99])
    model_aprox = SemiBatchReactor(k=[k1, k2, 0, 0, 5])
    input_signal = f_input
