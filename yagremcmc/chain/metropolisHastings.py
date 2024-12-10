from abc import ABC, abstractmethod
from numpy.random import uniform

from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.interface import ChainDiagnostics
from yagremcmc.chain.transition import TransitionData
from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.chain import Chain
from yagremcmc.utility.verbosity import VerbosityController


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
        self._verbosityController = VerbosityController()

    @property
    def chain(self):
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @property
    def diagnostics(self):
        return self._diagnostics

    def set_up_verbosity_controller(self, chainLength, verbose):

        if verbose:
            self._verbosityController.prepare(chainLength, self._diagnostics)
        else:
            self._verbosityController.turn_off()

    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state) -> TransitionData:

        # acceptance probability is zero, omit evaluation of the likelihood
        # in __acceptance_probability. The probability of this happening is
        # non-zero in MLDA
        if proposal == state:
            return TransitionData(state, proposal, TransitionData.REJECTED)

        acceptProb = self._acceptance_probability(proposal, state)

        if acceptProb < 0. or 1. < acceptProb:
            raise RuntimeError(f"invalid acceptance probability: {acceptProb}")

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:
            return TransitionData(state, proposal, TransitionData.ACCEPTED)
        else:
            return TransitionData(state, proposal, TransitionData.REJECTED)

    def _update_chain(self, nextState):
        self._chain.append(nextState.coefficient)

    # make it a class method to allows derived classed to override
    def determine_next_state(self, transitionData):

        nextState = None

        if transitionData.outcome == TransitionData.ACCEPTED:
            nextState = transitionData.proposal

        elif transitionData.outcome == TransitionData.REJECTED:
            nextState = transitionData.state
        else:
            raise ValueError(
                f"Invalid transition outcome: {transitionData.outcome}")

        return nextState

    def _process_transition(self, transitionData):

        self._diagnostics.process(transitionData)

        nextState = self.determine_next_state(transitionData)
        self._update_chain(nextState)

        return nextState

    def run(self, chainLength, initialState, verbose=True):

        self.set_up_verbosity_controller(chainLength, verbose)

        self._chain.clear()
        self._chain.append(initialState.coefficient)

        state = initialState.clone_with(self._chain.trajectory[0])

        for n in range(chainLength - 1):

            self._verbosityController.run(n)

            self._proposalMethod.set_state(state)
            proposal = self._proposalMethod.generate_proposal()

            transitionOutcome = self._accept_reject(proposal, state)
            state = self._process_transition(transitionOutcome)
