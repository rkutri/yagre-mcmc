import numpy as np

from yagremcmc.chain.proposal import ProposalMethod

# TODO: reuse structure from the asm proposal
class AdaptiveProposal(ProposalMethod):

    def __init__(self, initCov, adaptiveCov, idleSteps):
        """
        Parameters:
        - initCov: Initial covariance matrix to be used during the IDLE phase
        - adaptiveCov: Adaptive covariance matrix to be used after IDLE phase
        - idleSteps: Number of steps during which the covariance is not updated
                     (IDLE phase).
        """

        if initCov.dimension == 1:
            raise NotImplementedError("Adaptivity not implemented for scalar chains.")

        super().__init__()

        self._proposalMethod = MRWProposal(initCov)

        self.iSteps_ = idleSteps
        self.cSteps_ = collectionSteps


    def get_state(self):
        return self._proposalMethod.get_state()

    def set_state(self, newState):
        self._proposalMethod.set_state(newState)

    @property
    def chain(self):
        return self._chain

    @chain.setter
    def chain(self, newChain):
        self._chain = newChain

    def _determine_proposal_method(self):

        if self._chain.length < self.iSteps_ + self.cSteps_:
            assert isinstance(self._proposalMethod, MRWProposal)

        elif self._chain.length == self.iSteps_ + self.cSteps_:

            currentState = self._proposalMethod.get_state()

            self._proposalMethod = AMProposal(
                self._chain, self.eps_, self.cSteps_)
            self._proposalMethod.set_state(currentState)

            amLogger.info("Start adaptive covariance")

        elif self._chain.length > self.iSteps_ + self.cSteps_:
            assert isinstance(self._proposalMethod, AMProposal)

        else:
            raise RuntimeError("Undefined adaptive Metropolis chain state.")

    def generate_proposal(self):

        if self._chain is None:
            raise ValueError(
                "Adaptive Proposal is not associated with a chain yet.")

        self._determine_proposal_method()

        return self._proposalMethod.generate_proposal()


