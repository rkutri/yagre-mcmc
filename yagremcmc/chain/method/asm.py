import logging

import numpy as np

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings, UnnormalisedPosterior
from yagremcmc.chain.method.mrw import MRWProposal
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.statistics.shrinkage import ledoit_wolf_shrinkage
from yagremcmc.statistics.parameterLaw import Gaussian


asmLogger = logging.getLogger(__name__)
asmLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
consoleHandler.setFormatter(formatter)

asmLogger.addHandler(consoleHandler)


class ASMProposal(ProposalMethod):
    """
    Adaptive Shrinkage Metropolis proposals
    """

    def __init__(self, chain, startIdx):

        super().__init__()

        self._chain = chain
        self._sIdx = startIdx

        states = self._chain.accepted_states(sIdx)

        self._mean = np.mean(states, axis=0)
        self._nData = len(states)

        self._cStates = states - self._mean

        # assuming zero-mean
        self._empiricalCov = np.dot(self._cStates, self._cStates.T) \
            / (self._nData - 1)

        # FIXME
        self._shrinkageTarget = ...

        self._proposalLaw = None

        asmLogger.info(f"Adaptive covariance initialised using {self._nData}"
                       " states.")

    def set_state(self, newState):

        proposalCov = ledoit_wolf_shrinkage(self._empiricalCov,
                                            self._shrinkageTarget)

        self._state = newState
        self._proposalLaw = Gaussian(self._state, proposalCov)

    def generate_proposal(self):

        self._update_covariance()

        return self._proposalLaw.generate_realisation()

    def _update_covariance(self):

        states = self._chain.accepted_states(self._sIdx)

        if not self._update_required(states):
            return

        newState = states[-1]

        # TODO: update mean

        # TODO: update centred states

        # TODO: update empirical covariance

    def _update_required(self, acceptedStates):

        if self._nData == len(acceptedStates):
            return False

        else:

            if (len(acceptedStates) - self._nData == 1):
                return True
            else:
                raise RuntimeError("Inconsistent data size")
