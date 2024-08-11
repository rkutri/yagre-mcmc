from abc import ABC, abstractmethod


class ChainConfiguration(ABC):

    @abstractmethod
    def target_is_posterior(self):
        pass

    @abstractmethod
    def target_is_explicit(self):
        pass

    @abstractmethod
    def set_bayes_model(self, model):
        pass

    @abstractmethod
    def set_target_density(self, density):
        pass
