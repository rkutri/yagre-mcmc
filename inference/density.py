from inference.interface import DensityInterface


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self.model_ = model

    def evaluate_log(self, parameter):

        return self.model_.log_likelihood(parameter) \
            + self.model_.log_prior(parameter)


# template class
class DifferentiableDensity(DensityInterface):
    pass


