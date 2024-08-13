from abc import ABC, abstractmethod
from numpy.random import uniform
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.chain import Chain


class ChainDiagnostics:

    def __init__(self, chain) -> None:

        self.nAccept_ = 0
        self.chain_ = chain

    def add_accepted(self):

        self.nAccept_ += 1

    def acceptance_rate(self):

        # initial state does not count as accepted
        return self.nAccept_ / (len(self.chain_.trajectory) - 1)


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self, targetDensity: DensityInterface,
                 proposalMethod: ProposalMethodInterface) -> None:

        self.targetDensity_ = targetDensity
        self.proposalMethod_ = proposalMethod

        self.chain_ = Chain()
        self.diagnostics_ = ChainDiagnostics(self.chain_)

    @property
    def chain(self):
        return self.chain_

    @property
    def diagnostics(self):
        return self.diagnostics_

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state):

        acceptProb = self._acceptance_probability(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:

            self.diagnostics_.add_accepted()
            return proposal

        else:
            return state

    # TODO: switch to Python logging for verbosity
    def run(self, nSteps, initialState, verbose=True):

        state = initialState

        self.chain_.append(state.coefficient)

        for n in range(nSteps - 1):

            if verbose:
                if (n % 50 == 0):
                    if (n == 0):
                        print("Start Markov chain")
                    else:
                        print(str(n) + " steps computed")

            self.proposalMethod_.state = state
            proposal = self.proposalMethod_.generate_proposal()

            state = self._accept_reject(proposal, state)

            self.chain_.append(state.coefficient)

        return
