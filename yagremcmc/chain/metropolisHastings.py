from abc import ABC, abstractmethod
from numpy.random import uniform
from yagremcmc.statistics.interface import DensityInterface


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self, targetDensity: DensityInterface) -> None:

        self.states_ = []
        self.targetDensity_ = targetDensity

    @abstractmethod
    def generate_proposal__(self, state):
        pass

    @abstractmethod
    def acceptance_probability__(self, proposal, state):
        pass

    def accept_reject__(self, proposal, state):

        acceptProb = self.acceptance_probability__(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)

        if decision <= acceptProb:
            return proposal
        else:
            return state

    @property
    def states(self):
        return self.states_

    def run(self, nSteps, initialState, verbose=True):

        state = initialState

        self.states_ = [state.coefficient]

        for n in range(nSteps - 1):

            if verbose:
                if (n % 50 == 0):
                    if (n == 0):
                        print("Start Markov chain")
                    else:
                        print(str(n) + " steps computed")

            proposal = self.generate_proposal__(state)

            state = self.accept_reject__(proposal, state)

            self.states_.append(state.coefficient)

        return
