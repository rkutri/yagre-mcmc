from numpy import exp
from inference.interface import LikelihoodInterface


class BayesianRegressionLikelihood(LikelihoodInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self.data_ = data
        self.forwardModel_ = forwardModel
        self.noiseModel_ = noiseModel

    def evaluate_log_likelihood(self, parameter):

        modelResponse = self.forwardModel_.evaluate(parameter)

        dataMisfit = modelResponse - self.data_.array

        logL = 0.

        for i in range(self.data_.size):

            li = -0.5 * self.noiseModel_.induced_norm_squared(dataMisfit[i, :])

            logL += li

        return logL
