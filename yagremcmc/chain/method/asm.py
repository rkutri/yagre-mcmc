import logging

import numpy as np

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.chain.method.mrw import MRWProposal
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.statistics.regularisation import regularised_marginal_variance_weights
from yagremcmc.statistics.parameterLaw import Gaussian


asmLogger = logging.getLogger(__name__)
asmLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
consoleHandler.setFormatter(formatter)

asmLogger.addHandler(consoleHandler)

# TODO: Refactor to implement the covariance interface instead of 
#       a proposal method interface
class AdaptiveWeightProposal(ProposalMethod):

    def __init__(self, chain, initCov, startIdx):

        super().__init__()

        self._chain = chain
        self._cov = initCov
        self._sIdx = startIdx

        states = self._chain.accepted_states(startIdx)

        self._nData = len(states)

        self._empMargVar = np.var(states, axis=0)

        self._proposalLaw = None

        asmLogger.info(f"Adaptive covariance initialised using {self._nData}"
                       " states.")

    def set_state(self, newState):

        self._update_marginal_variance()

        weights = regularised_marginal_variance_weights(self._empMargVar)

        self._cov.reweight_dimensions(weights)

        self._state = newState
        self._proposalLaw = Gaussian(self._state, self._cov)

    def generate_proposal(self):
        return self._proposalLaw.generate_realisation()

    def _update_marginal_variance(self):

        states = self._chain.accepted_states(self._sIdx)

        if not self._update_required(states):
            return

        self._empMargVar = np.var(states, axis=0)

        self._nData = len(states)

    def _update_required(self, acceptedStates):

        if self._nData == len(acceptedStates):
            return False

        else:

            if (len(acceptedStates) - self._nData == 1):
                return True
            else:
                raise RuntimeError("Inconsistent data size")


class AdaptiveProposal(ProposalMethod):
    """
        Adaptive Rescaling Metropolis Proposal
    """

    def __init__(self, initCov, idleSteps, collectionSteps)