from numpy import dot
from inference.interface import NoiseModelInterface


class CentredGaussianIIDNoise(NoiseModelInterface):

    def __init__(self, variance):

        self.variance_ = variance

    def induced_norm_squared(self, vector):
        """
        vector: np.ndarray
        """

        return dot(vector, vector) / self.variance_
