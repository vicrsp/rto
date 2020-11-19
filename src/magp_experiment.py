import numpy as np
import multiprocessing

from optimization.rto import RTO
from optimization.batch_profile_optimizer import BatchProfileOptimizer
from model.adaptation.ma_gaussian_processes import MAGaussianProcesses
from model.process.semi_batch import SemiBatchReactor
from model.utils import generate_samples_uniform

n_experiments = 20
n_iterations = 120
data_size = 5

model = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
plant = SemiBatchReactor()
u_real_optimum = [18.4427644, 0.00110823777, 227.792418]
u_0 = [10.652103265931729, 0.0005141834799295323, 224.48063936756103]
g_plant = np.array([0.025, 0.15])

def run_rto_exact():
    # Exact algorithm
    for i in range(n_experiments):
        print('slsqp_scipy experiment {}'.format(i))
        initial_data = generate_samples_uniform(
            model, plant, g_plant, u_0, data_size)
        opt_problem = BatchProfileOptimizer(
            ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='slsqp_scipy')

        adaptation = MAGaussianProcesses(model, initial_data)
        u_0_feas = initial_data[0][0]
        rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
                  db_file='/mnt/d/rto_data/rto_poc_exact_slsqp.db', name='ma-gp-slsqp_scipy_last10')
        rto.run(u_0_feas)

def run_rto_de_best1bin():
    # DE best1bin algorithm
    for i in range(n_experiments):
        print('de_scipy_best1bin experiment {}'.format(i))
        initial_data = generate_samples_uniform(
            model, plant, g_plant, u_0, data_size)
        opt_problem = BatchProfileOptimizer(
            ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_best1bin')

        adaptation = MAGaussianProcesses(model, initial_data)

        rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
                  db_file='/mnt/d/rto_data/rto_poc_debest1bin.db', name='ma-gp-de_scipy_best1bin_last10')
        rto.run(u_0)


def run_rto_de_rand1bin():
    # DE rand1bin algorithm
    for i in range(n_experiments):
        print('de_scipy_rand1bin experiment {}'.format(i))
        initial_data = generate_samples_uniform(
            model, plant, g_plant, u_0, data_size)
        opt_problem = BatchProfileOptimizer(
            ub=[30, 0.002, 250], lb=[0, 0, 200], g=[0.025, 0.15], solver='de_scipy_rand1bin')

        adaptation = MAGaussianProcesses(model, initial_data)

        rto = RTO(model, plant, opt_problem, adaptation, iterations=n_iterations,
                  db_file='/mnt/d/rto_data/rto_poc_derand1bin.db', name='ma-gp-de_scipy_rand1bin_last10')
        rto.run(u_0)

if __name__ == '__main__':
    # Define general constants
    p1 = multiprocessing.Process(target=run_rto_exact)
    p1.start()
    p2 = multiprocessing.Process(target=run_rto_de_best1bin)
    p2.start()
    p3 = multiprocessing.Process(target=run_rto_de_rand1bin)
    p3.start()
    jobs = [p1,p2,p3]
    [job.join() for job in jobs]
    #run_rto_pso()
