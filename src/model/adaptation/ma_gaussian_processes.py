import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter


class MAGaussianProcesses:
    def __init__(self, process_model):
        self.process_model = process_model
        self.u_k = []
        self.samples_k = []
        self.models = None

    def train(self, X, y):
        gp_model = GaussianProcessRegressor()
        return gp_model.fit(X, y)

    def get_modifiers(self, u):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return np.asarray([model.predict(u, return_std=False) for model in self.models])
    
    def get_model_parameters(self):
        return self.process_model.initial_parameters

    def adapt(self, u, samples):
        # update the data available
        self.u_k.append(u)
        self.samples_k.append(samples)
        # get the data that will be used for training
        # TODO: limit amount of data used to train
        u_train = np.asarray(self.u_k)
        y_train = np.asarray(self.samples_k)
        _, cols = y_train.shape
        
        # train the model for objective modifiers
        models = []
        models.append(self.train(u_train, y_train[:, 0]))

        # train the model for constraints modifiers
        for col in range(1, cols):
            models.append(self.train(u_train, y_train[:, col]))

        self.models = models
