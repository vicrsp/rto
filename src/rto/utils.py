import numpy as np
from smt.sampling_methods import LHS



def generate_samples_uniform(model, plant, plant_constrains, u_0, size=30, offset=0.8, noise=0.01):
    def rand_func(x):
        scale = (2 * (1 - offset))
        return x * (np.random.rand(len(x)) * scale + offset)
    return generate_samples(model, plant, plant_constrains, u_0, rand_func, size, noise)


def generate_samples_gaussian(model, plant, plant_constrains, u_0, size=30, scale=0.01):
    def rand_func(x):
        return np.asarray([x_i * (1 + np.random.normal(scale=scale)) for x_i in x])
    return generate_samples(model, plant, plant_constrains, u_0, rand_func, size)


def generate_samples_lhs(model, plant, plant_constrains, u_0, size=30, offset=0.2, noise=0.01, criterion='maximin'):
    xlimits = np.vstack((np.array(u_0)*(1-offset), np.array(u_0)*(1+offset))).T
    sampling = LHS(xlimits=xlimits, criterion=criterion)

    samples = sampling(size)
    u_initial = []
    y_initial = []
    meas_initial = []

    for sample in samples:
        fr, gr = plant.get_objective(sample, noise=noise), plant.get_constraints(sample, noise=noise)
        fm, gm = model.get_objective(sample), model.get_constraints(sample)
        u_initial.append(sample)
        y_initial.append(np.append(fr - fm, gr - gm))
        meas_initial.append(np.array([fr, gr, fm, gm], dtype=object))
    
    return [np.asarray(u_initial), np.asarray(y_initial), np.asarray(meas_initial)]
    # return generate_samples(model, plant, plant_constrains, u_0, rand_func, size)
        
def generate_samples(model, plant, plant_constrains, u_0, random_func, size, noise = 0.01):
    u_initial = []
    y_initial = []
    meas_initial = []
    
    i = 0
    while(i < size):
        u_rand = random_func(u_0)
        fr, gr = plant.get_objective(
            u_rand, noise=noise), plant.get_constraints(u_rand, noise=noise)

        fm, gm = model.get_objective(
            u_rand), model.get_constraints(u_rand)

        # append only if constraints are not violated
        if(np.any(gr - plant_constrains > 0) == False):
            u_initial.append(u_rand)
            y_initial.append(np.append(fr - fm, gr - gm))
            meas_initial.append(np.array([fr, gr, fm, gm]))
            i = i + 1

    return [np.asarray(u_initial), np.asarray(y_initial), np.asarray(meas_initial)]
