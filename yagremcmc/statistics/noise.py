from numpy import dot
from yagremcmc.statistics.interface import NoiseModelInterface


class CentredGaussianNoise(NoiseModelInterface):

    @property
    def covariance(self):
        pass


class IndependentGaussianSumNoise(CentredGaussianNoise):
    pass


class CentredGaussianIIDNoise(CentredGaussianNoise):

    def __init__(self, variance):

        self.variance_ = variance

    def induced_norm_squared(self, vector):
        """
        vector: np.ndarray
        """

        return dot(vector, vector) / self.variance_
