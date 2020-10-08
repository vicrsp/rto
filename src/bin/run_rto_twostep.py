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

from optimization.de import DifferentialEvolution
from optimization.model_optimization import ProfileOptimizer, ModelParameterOptimizer
from model.semi_batch import SemiBatchReactor
from optimization.utils import build_F

## Get optimal profiles for initial set of params
opt = ProfileOptimizer()
model_aprox = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
f_aprox, x_aprox, results_aprox = opt.run(model_aprox)

## Calibrate model parameters
cal = ModelParameterOptimizer()
model_ideal = SemiBatchReactor()

samples = model_ideal.get_samples(x_aprox, [0.95,0.97,0.99])
f_cal, x_cal, results_cal = cal.run(model_aprox, x_aprox, samples)
print('Fcal---> {}'.format(f_cal))
print('Xcal---> {}'.format(x_cal))

