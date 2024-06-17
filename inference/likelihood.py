from numpy import exp
from inference.interface import LikelihoodInterface


class BayesianRegressionLikelihood(LikelihoodInterface):

    def __init__(self, measurements, forwardModel, noiseModel):

        self.data_ = measurements
        self.forwardModel_ = forwardModel
        self.noiseModel_ = noiseModel

    def evaluate_log_likelihood(self, parameter):

        modelResponse = self.forwardModel_.evaluate(parameter)

        logL = 0.

        for i in range(self.data_.size):

            dataMisfit = modelResponse[i] - self.data_.measurement[i]

            li = -0.5 * self.noiseModel_.induced_norm_squared(dataMisfit)

            logL += li

        return logL
