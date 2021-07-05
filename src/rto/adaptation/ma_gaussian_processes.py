import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from warnings import catch_warnings
from warnings import simplefilter

from base import AdaptationResult, AdaptationStrategy

class MAGaussianProcesses(AdaptationStrategy):
    def __init__(self, process_model, initial_data, neighbors_type='k_last', k_neighbors=10, filter_data=True):
        super().__init__(process_model, initial_data)
        self.u_k = []
        self.samples_k = []
        self.models = None
        self.initialize_models(self.initial_data)
        self.k_neighbors = k_neighbors
        self.neighbors_type = neighbors_type
        self.filter_data = filter_data

    def get_adaptation(self, u):
        return AdaptationResult('ma', {'modifiers': self.get_modifiers(u)})

    def initialize_models(self, data):
        u_train, y_train, _ = data
        self.u_k = list(u_train)
        self.samples_k = list(y_train)
        self.update_gp_model(u_train, y_train)

    def update_gp_model(self, X, y):
        _, cols = y.shape

        # normalization
        self.update_normalization_params(X, y)
        X_norm = self.normalize_input(X)

        # train the GP model
        models = []
        for col in range(cols):
            models.append(self.train(X_norm, self.normalize_output(y, col)))

        self.models = models

    def normalize_input(self, x):
        return self.input_scaler.transform(x)

    def normalize_output(self, y, index):
        return self.output_scalers[index].transform(y[:, index].reshape(-1, 1))

    def denormalize_output(self, y, index):
        return self.output_scalers[index].inverse_transform(y).flatten()

    def update_normalization_params(self, inputs, outputs):
        _, cols = outputs.shape
        self.input_scaler = StandardScaler().fit(inputs)
        self.output_scalers = [StandardScaler().fit(
            outputs[:, col].reshape(-1, 1)) for col in range(cols)]

    def train(self, X, y):
        gp_model = GaussianProcessRegressor()
        return gp_model.fit(X, y)

    def get_modifiers(self, u):
        # normalize
        u_norm = self.normalize_input(u.reshape(1, -1))
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return np.asarray([self.denormalize_output(model.predict(u_norm), index) for index, model in enumerate(self.models)])
    
    def get_model_parameters(self):
        return self.process_model.initial_parameters

    def adapt(self, u, samples):
        data_size = len(self.u_k)
        neighbors_size = min(self.k_neighbors, data_size)

        if(self.neighbors_type == 'k_last'):  # use data from the k last operating points
            u_train = np.asarray(self.u_k[-neighbors_size:])
            y_train = np.asarray(self.samples_k[-neighbors_size:])
        elif(self.neighbors_type == 'k_nearest'):  # use data from the k nearest operating points
            # scale the input data to [0,1] interval
            scaler = MinMaxScaler()
            u_norm = scaler.fit_transform(np.asarray(self.u_k))
            # find the neighbors
            nbrs = NearestNeighbors(
                n_neighbors=neighbors_size, algorithm='ball_tree').fit(u_norm)
            _, indices = nbrs.kneighbors(
                scaler.transform(u.reshape(1, -1)))

            if(data_size > self.k_neighbors):
                u_train = np.asarray(self.u_k)[indices.flatten(), :]
                y_train = np.asarray(self.samples_k)[indices.flatten(), :]
            else:
                u_train = np.asarray(self.u_k)
                y_train = np.asarray(self.samples_k)

        self.update_gp_model(u_train, y_train)
        if(self.filter_data == True):
            # if filter is on, only append new data to the model is sufficiently far from
            scaler = MinMaxScaler()
            u_norm = scaler.fit_transform(np.asarray(self.u_k))
            nbrs = NearestNeighbors(
                n_neighbors=neighbors_size, algorithm='ball_tree').fit(u_norm)
            distances, _ = nbrs.kneighbors(scaler.transform(u.reshape(1, -1)))
            # if there is at least one operating point below the threshold
            # then we should be able to ignore the new data
            if(np.all(distances > 0.01)):
                self.u_k.append(u)
                self.samples_k.append(samples)

        else:
            self.u_k.append(u)
            self.samples_k.append(samples)
