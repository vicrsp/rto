import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter


class MAGaussianProcesses:
    def __init__(self, process_model, initial_data):
        self.process_model = process_model
        self.u_k = []
        self.samples_k = []
        self.models = None
        self.initialize_models(initial_data)

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

    def get_normalization_params(self, X):
        return np.mean(X, axis=0), np.std(X, axis=0)

    def normalize_input(self, x):
        u_mean, u_std = self.input_normalization_params
        return (x - u_mean) / u_std

    def normalize_output(self, y, index):
        mean, std = self.output_normalization_params[index]
        return (y[:, index] - mean) / std

    def denormalize_output(self, y, index):
        mean, std = self.output_normalization_params[index]
        return y * std + mean

    def update_normalization_params(self, inputs, outputs):
        _, cols = outputs.shape
        self.input_normalization_params = self.get_normalization_params(inputs)
        self.output_normalization_params = [
            self.get_normalization_params(outputs[:, col]) for col in range(cols)]

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
        # TODO: limit amount of data used to train
        u_train = np.asarray(self.u_k[-30:])
        y_train = np.asarray(self.samples_k[-30:])

        self.update_gp_model(u_train, y_train)
