from numpy import exp
from inference.interface import DensityInterface

from inference.data import Data
import numpy as np


class BayesianRegressionLikelihood(DensityInterface):

    def __init__(self, data, forwardModel, noiseModel):

        self.data_ = data
        self.fwdModel_ = forwardModel
        self.noiseModel_ = noiseModel

    def evaluate_log(self, parameter):

        dataMisfit = self.fwdModel_.evaluate(parameter) - self.data_.array

        dmNormSquared = np.apply_along_axis(
            lambda x: self.noiseModel_.induced_norm_squared(x), 1, dataMisfit)

        logL = -0.5 * np.sum(dmNormSquared)

        return logL
