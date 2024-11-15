from abc import abstractmethod
from yagremcmc.statistics.interface import ParameterLawInterface
from yagremcmc.parameter.interface import ParameterInterface


class AbsolutelyContinuousParameterLaw(ParameterLawInterface):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def density(self):
        pass
