from abc import ABC, abstractmethod
from numpy.random import uniform
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.chain import Chain


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self, targetDensity: DensityInterface,
                 proposalMethod: ProposalMethodInterface) -> None:

        self.chain_ = Chain()
        self.targetDensity_ = targetDensity
        self.proposalMethod_ = proposalMethod

    @property
    def chain(self):
        return self.chain_

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state):

        acceptProb = self._acceptance_probability(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)

        if decision[0] <= acceptProb:
            return (proposal, True)
        else:
            return (state, False)

    # TODO: switch to Python logging for verbosity
    def run(self, nSteps, initialState, verbose=True):

        state = initialState

        self.chain_.add_state_vector(state.coefficient, False)

        for n in range(nSteps - 1):

            if verbose:
                if (n % 50 == 0):
                    if (n == 0):
                        print("Start Markov chain")
                    else:
                        print(str(n) + " steps computed")

            self.proposalMethod_.state = state
            proposal = self.proposalMethod_.generate_proposal()

            state, isAccepted = self._accept_reject(proposal, state)

            self.chain_.add_state_vector(state.coefficient, isAccepted)

        return
