from abc import ABC, abstractmethod
from numpy.random import uniform

from yagremcmc.utility.boilerplate import create_logger
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.interface import ChainDiagnostics
from yagremcmc.chain.transition import TransitionData
from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.chain import Chain


mhLogger = create_logger()


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self,
                 targetDensity: DensityInterface,
                 proposalMethod: ProposalMethod,
                 diagnostics: ChainDiagnostics
                 ) -> None:

        super().__init__()

        self._tgtDensity = targetDensity
        self._proposalMethod = proposalMethod
        self._diagnostics = diagnostics

        self._chain = Chain()

    @property
    def chain(self):
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @property
    def diagnostics(self):
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, diagnostics):
        self._diagnostics = diagnostics

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state) -> TransitionData:

        # acceptance probability is zero, omit evaluation of the likelihood
        # in __acceptance_probability. The probability of this happening is
        # non-zero in MLDA
        if proposal == state:
            return TransitionData(state, TransitionData.REJECTED)

        acceptProb = self._acceptance_probability(proposal, state)

        if acceptProb < 0. or 1. < acceptProb:
            raise RuntimeError(f"invalid acceptance probability: {acceptProb}")

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:
            return TransitionData(proposal, TransitionData.ACCEPTED)
        else:
            return TransitionData(state, TransitionData.REJECTED)

    def run(self, chainLength, initialState, verbose=True,
            nPrintIntervals=20, minInterval=10):

        self._chain.clear()
        self._chain.append(initialState.coefficient)

        state = initialState.clone_with(self._chain.trajectory[0])

        for n in range(chainLength - 1):

            if verbose:

                if (n == 0):
                    mhLogger.info("Start Markov chain")

                interval = max(chainLength // nPrintIntervals, minInterval)
                self._diagnostics.lag = interval

                if (n % interval == 0 and n > 0):

                    mhLogger.info(
                        f"{n} steps computed. Calculating diagnostics.")

                    self._diagnostics.print_diagnostics(mhLogger)

            self._proposalMethod.set_state(state)
            proposal = self._proposalMethod.generate_proposal()

            transitionOutcome = self._accept_reject(proposal, state)
            state = transitionOutcome.state

            self._diagnostics.process(transitionOutcome)
            self._chain.append(state.coefficient)

        return
