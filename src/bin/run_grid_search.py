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

from optimization.grid_search import ParameterGridSearch
from optimization.model_optimization import ProfileOptimizer
from model.semi_batch import SemiBatchReactor

gs = ParameterGridSearch(lb=[0.15, 0.15], ub=[0.5, 0.6], N = 10)
model_ideal = SemiBatchReactor()
model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])

# Get the fixed model inputs
# TODO: read from DB
opt = ProfileOptimizer()
_, F, _ = opt.Run(model)
samples = model_ideal.GetSamples(F, [0.95,0.97,0.99], noise=False)
samples_noisy = model_ideal.GetSamples(F, [0.95,0.97,0.99], noise=True)

grid, results, pareto = gs.run(model, F, samples)
grid_noisy, results_noisy, pareto_noisy = gs.run(model, F, samples_noisy)

# Plot the grid search results
for key in grid.keys():
    fig = plt.figure()
    plt.subplot(121)
    X, Y = grid[key]
    Z = results[key]
    cs = plt.contourf(X,Y,Z,cmap='jet')
    fig.colorbar(cs)
    
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot_wireframe(X,Y,Z)

    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(30, 30)
    ax.plot_surface(X,Y,Z,cmap='jet')

plt.show()

# Compare the noisy and ideal surfaces
for key in grid.keys():
    fig = plt.figure()
    plt.subplot(121)
    X, Y = grid[key]
    
    Z = results[key]
    Z_noisy = results_noisy[key]

    cs = plt.contourf(X,Y,Z,cmap='jet')
    csn = plt.contour(X,Y,Z_noisy,cmap='jet')
    fig.colorbar(cs)
    fig.colorbar(csn)
    
    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot_wireframe(X,Y,Z)

    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(30, 30)
    ax.plot_surface(X,Y,Z,cmap='jet')
    ax.plot_wireframe(X,Y,Z_noisy)

plt.show()