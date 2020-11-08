import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from warnings import catch_warnings
from warnings import simplefilter


class MAGaussianProcesses:
    def __init__(self, process_model, initial_data):
        self.process_model = process_model
        self.u_k = []
        self.samples_k = []
        self.models = None
        self.initialize_models(initial_data)
        self.k_neighbors = 20

    def initialize_models(self, data):
        u_train, y_train = data
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
        # update the data available
        self.u_k.append(u)
        self.samples_k.append(samples)
        # get the data that will be used for training
        # from the k-nearest neighbors
        # TODO: find neighbors excluding last sample?
        if(len(self.u_k) > self.k_neighbors):
            scaler = MinMaxScaler()
            u_norm = scaler.fit_transform(np.asarray(self.u_k))
            nbrs = NearestNeighbors(
                n_neighbors=self.k_neighbors, algorithm='ball_tree').fit(u_norm)
            _, indices = nbrs.kneighbors(scaler.transform(u.reshape(1, -1)))

            # TODO: filter points based on distance?
            u_train = np.asarray(self.u_k)[indices.flatten(), :]
            y_train = np.asarray(self.samples_k)[indices.flatten(), :]
        else:
            u_train = np.asarray(self.u_k)
            y_train = np.asarray(self.samples_k)

        self.update_gp_model(u_train, y_train)
