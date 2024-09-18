import logging

from abc import ABC, abstractmethod
from numpy.random import uniform

from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.chain import Chain
from yagremcmc.chain.diagnostics import ChainDiagnostics

# Set up the logger
mhLogger = logging.getLogger(__name__)
mhLogger.setLevel(logging.INFO)

# Create a console handler
consoleHandler = logging.StreamHandler()

# Set a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(levelname)s: %(message)s')
consoleHandler.setFormatter(formatter)

# Add the console handler to the mhLogger
mhLogger.addHandler(consoleHandler)

class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self, targetDensity: DensityInterface,
                 proposalMethod: ProposalMethod) -> None:

        super().__init__()

        self._tgtDensity = targetDensity
        self._proposalMethod = proposalMethod

        self._chain = Chain()
        self._diagnostics = ChainDiagnostics(self._chain)

    @property
    def chain(self):
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @property
    def diagnostics(self):
        return self._diagnostics

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state):

        acceptProb = self._acceptance_probability(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:

            self._diagnostics.add_accepted()
            return proposal

        else:
            self._diagnostics.add_rejected()
            return state

    def run(self, nSteps, initialState, verbose=True):

        self._chain.clear()
        self._chain.append(initialState.coefficient)

        state = initialState

        for n in range(nSteps - 1):

            if verbose:
                interval = nSteps // 20
                if (n % interval == 0):
                    if (n == 0):
                        mhLogger.info("Start Markov chain")
                    else:
                        ra = self._diagnostics.rolling_acceptance_rate(
                            interval)
                        mhLogger.info(f"{n} steps computed")
                        mhLogger.info(f"  - rolling acceptance rate: {ra}")

            self._proposalMethod.set_state(state)
            proposal = self._proposalMethod.generate_proposal()

            state = self._accept_reject(proposal, state)

            self._chain.append(state.coefficient)

        return
