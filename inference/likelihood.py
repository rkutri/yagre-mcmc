from numpy import exp
from inference.interface import LikelihoodInterface


class BayesianRegressionLikelihood(LikelihoodInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self.data_ = data
        self.forwardModel_ = forwardModel
        self.noiseModel_ = noiseModel

    def evaluate_log_likelihood(self, parameter):

        logL = 0.

        modelResponse = self.forwardModel_.evaluate(parameter)

        for i in range(self.data_.size):

            dataMisfit = modelResponse[i] - self.data_.measurement[i]

            li = -0.5 * self.noiseModel_.induced_norm_squared(dataMisfit)

            logL += li

        return logL
