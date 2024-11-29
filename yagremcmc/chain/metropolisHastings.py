from abc import ABC, abstractmethod
from enum import IntEnum
from numpy.random import uniform

from yagremcmc.utility.boilerplate import create_logger
from yagremcmc.chain.diagnostics import print_diagnostics
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.chain import Chain


mhLogger = create_logger()


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    proposalRejected = 0
    proposalAccepted = 1

    def __init__(self, targetDensity: DensityInterface,
                 proposalMethod: ProposalMethod) -> None:

        super().__init__()

        self._tgtDensity = targetDensity
        self._proposalMethod = proposalMethod

        self._chain = Chain()

    @property
    def chain(self):
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state):

        # acceptance probability is zero, omit evaluation of the likelihood
        # in __acceptance_probability. The probability of this happening is
        # non-zero in MLDA
        if proposal == state:
            return state, MetropolisHastings.proposalRejected

        acceptProb = self._acceptance_probability(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:
            return proposal, MetropolisHastings.proposalAccepted
        else:
            return state, MetropolisHastings.proposalRejected

    def run(self, chainLength, initialState, verbose=True,
            nPrintIntervals=20, minInterval=10):

        self._chain.clear_trajectory()
        self._chain.append(initialState.coefficient, True)

        state = initialState.clone_with(self._chain.trajectory[0])

        for n in range(chainLength - 1):

            if verbose:

                if (n == 0):
                    mhLogger.info("Start Markov chain")

                interval = max(chainLength // nPrintIntervals, minInterval)

                if (n % interval == 0 and n > 0):

                    mhLogger.info(
                        f"{n} steps computed. Calculating diagnostics.")

                    print_diagnostics(
                        mhLogger, self._chain.diagnostics, interval)

            self._proposalMethod.set_state(state)
            proposal = self._proposalMethod.generate_proposal()

            state, isAccepted = self._accept_reject(proposal, state)

            self._chain.append(state.coefficient, isAccepted)

        return
