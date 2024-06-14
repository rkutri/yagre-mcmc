from numpy import exp
from inference.interface import LikelihoodInterface


class BayesianRegressionLikelihood(LikelihoodInterface):

    def __init__(self, data, forwardMap, noiseModel):

        self.data_ = data
        self.forwardMap_ = forwardMap
        self.noiseModel_ = noiseModel

    def evaluate_log_likelihood(self, parameter):

        self.forwardMap_.parameter = parameter

        logL = 0.

        for i in range(self.data_.size):

            dataMisfit = self.forwardMap_.evaluate(self.data_.design[i])
            dataMisfit -= self.data_.measurement[i]

            li = -0.5 * self.noiseModel_.induced_norm_squared(dataMisfit)

            logL += li

        return logL
