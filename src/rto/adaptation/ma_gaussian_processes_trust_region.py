import numpy as np
from .ma_gaussian_processes import MAGaussianProcesses
from .base import AdaptationResult

class MAGaussianProcessesTrustRegion(MAGaussianProcesses):
    def __init__(self, process_model, initial_data, neighbors_type='k_last', k_neighbors=10, filter_data=True):
        super().__init__(process_model, initial_data, neighbors_type, k_neighbors, filter_data)
        self.tr_eta = [0.1, 0.1, 0.9]
        self.tr_t = [0.5, 2.0]
        self.tr_radii_max = 1
        self.tr_radii_start = 0.9
        self.raddi_k = [self.tr_radii_start]

    def get_adaptation(self, u):
        return AdaptationResult({'modifiers': self.get_modifiers(u), 'trust_region_radius': self.raddi_k[-1]})

    def calculate_cost_reduction_ratio(self, u, d, u_cost, d_cost):
        u_modifiers = self.get_modifiers(u)
        d_modifiers = self.get_modifiers(u + d)

        return (u_cost - d_cost)/(float(u_modifiers[0]) - float(d_modifiers[0]))

    def adapt(self, u, samples):
        data_size = len(self.u_k)
        neighbors_size = min(self.k_neighbors, data_size)
        u_train, y_train = self.get_training_data(u, data_size, neighbors_size)
        self.update_gp_model(u_train, y_train)
        self.update_gp_data(u, samples, neighbors_size)
    
    def update_operating_point(self, u, samples):
        u_k = self.u_k[-1]
        d_k = u - u_k

        d_cost = samples[0]
        u_k_cost = self.samples_k[-1][0]

        eta1, eta2, eta3 = self.tr_eta
        cost_ratio = self.calculate_cost_reduction_ratio(u_k, d_k, u_k_cost, d_cost)
       
        # update the trust region radius
        self.update_trust_region_radius(cost_ratio, eta2, eta3, d_k)
        # # update the operating point
        if(np.any(samples[1:] > 0)|(cost_ratio < eta1)):
            return u_k
        else:
            return u

    def update_trust_region_radius(self, cost_ratio, eta2, eta3, d_k):
        raddi = self.raddi_k[-1]
        t1, t2 = self.tr_t
        if(cost_ratio < eta2):
            raddi = t1 * raddi
        elif (cost_ratio > eta3)&(np.linalg.norm(d_k) == raddi):
            raddi = max(t2 * raddi, self.tr_radii_max)
        
        self.raddi_k.append(raddi)