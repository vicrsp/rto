import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from warnings import catch_warnings
from warnings import simplefilter
from .base import AdaptationResult, AdaptationStrategy

class MAGaussianProcesses(AdaptationStrategy):
    def __init__(self, process_model, initial_data, ub, lb, neighbors_type='k_last', k_neighbors=10, filter_data=True):
        super().__init__(process_model, initial_data, 'modifier_adaptation', ub, lb)
        self.u_k = []
        self.samples_k = []
        self.models = None
        self.initialize_models(self.initial_data)
        self.k_neighbors = k_neighbors
        self.neighbors_type = neighbors_type
        self.filter_data = filter_data
        self.filter_data_threshold = 0.01

    def get_adaptation(self, u, return_std=False):
        return AdaptationResult({'modifiers': self.get_modifiers(u, return_std)})

    def initialize_models(self, data):
        u_train, y_train = data.u, data.y

        self.u_k = list(u_train)
        self.samples_k = list(y_train)
        self.update_gp_model(u_train, y_train)

    def update_gp_model(self, X, y):
        _, cols = y.shape

        # normalization
        self.update_normalization_params(X, y)
        X_norm = self.normalize_model_input(X)

        # train the GP model
        models = []
        for col in range(cols):
            models.append(self.train(X_norm, y[:, col].reshape(-1, 1)))

        self.models = models

    def normalize_model_input(self, x):
        return self.input_scaler.transform(x)

    def normalize_model_output(self, y, index):
        return self.output_scalers[index].transform(y[:, index].reshape(-1, 1))

    def denormalize_model_output(self, y, index):
        return self.output_scalers[index].inverse_transform(y).flatten()

    def update_normalization_params(self, inputs, outputs):
        _, cols = outputs.shape
        self.input_scaler = StandardScaler().fit(inputs)
        self.output_scalers = [StandardScaler().fit(
            outputs[:, col].reshape(-1, 1)) for col in range(cols)]
    
    def train(self, X, y):
        gp_model = GaussianProcessRegressor(normalize_y=True)
        return gp_model.fit(X, y)

    def get_modifiers(self, u, return_std=True):
        # Then normalize to the model input space
        u_norm_model = self.normalize_model_input(u)
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return np.asarray([model.predict(u_norm_model,return_std=return_std) for model in self.models])
    
    def adapt(self, u, samples):
        data_size = len(self.u_k)
        neighbors_size = min(self.k_neighbors, data_size)
        u_train, y_train = self.get_training_data(u, data_size, neighbors_size)
        
        self.update_gp_model(u_train, y_train)
        self.update_gp_data(u, samples, neighbors_size)

    def update_gp_data(self, u, samples, neighbors_size):
        if(self.filter_data == True):
            # if filter is on, only append new data to the model is sufficiently far
            X = np.array(self.u_k)
            nbrs = NearestNeighbors(
                n_neighbors=neighbors_size, algorithm='ball_tree').fit(X)
            distances, _ = nbrs.kneighbors(u.reshape(1,-1))
            # if there is at least one operating point below the threshold
            # then we should be able to ignore the new data
            valid_distances = distances > self.filter_data_threshold
            if(np.all(valid_distances)):
                self.u_k.append(u)
                self.samples_k.append(samples)
        else:
            self.u_k.append(u)
            self.samples_k.append(samples)

    def get_training_data(self, u, data_size, neighbors_size):
        if(self.neighbors_type == 'k_last'):  # use data from the k last operating points
            u_train = np.asarray(self.u_k[-neighbors_size:])
            y_train = np.asarray(self.samples_k[-neighbors_size:])
        elif(self.neighbors_type == 'k_nearest'):  # use data from the k nearest operating points
            # scale the input data to [0,1] interval
            u_norm = np.asarray(self.u_k)
            # find the neighbors
            nbrs = NearestNeighbors(
                n_neighbors=neighbors_size, algorithm='ball_tree').fit(u_norm)
            _, indices = nbrs.kneighbors(u)

            if(data_size > self.k_neighbors):
                u_train = np.asarray(self.u_k)[indices.flatten(), :]
                y_train = np.asarray(self.samples_k)[indices.flatten(), :]
            else:
                u_train = np.asarray(self.u_k)
                y_train = np.asarray(self.samples_k)
        return u_train,y_train

