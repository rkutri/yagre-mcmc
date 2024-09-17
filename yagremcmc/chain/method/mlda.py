from numpy import exp

from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk


class SurrogateTransitionProposal(ProposalMethod, MetropolisHastings):

    def __init__(self, targetDensity, proposalMethod, subchainLength):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of surrogate transiton needs"
                             " to be derived from MetropolisHastings")

        MetropolisHastings.__init__(self, targetDensity, proposalMethod)
        ProposalMethod.__init__(self)

        self._subchainLength = subchainLength

    def generate_proposal(self):

        if self.state_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        # proposal of surrogate transition is itself a Metropolis-Hastings
        # algorithm
        self._proposalMethod.run(self._subchainLength, self._state)

        return self._proposalMethod.trajectory[-1]

    def _acceptance_probability(self, proposal, state):

        return exp(self._targetDensity.evaluate_log(proposal)
                   + self._proposalMethod.target.evaluate_log(state)
                   - self._proposalMethod.target.evaluate_log(proposal)
                   - self._targetDensity.evaluate_log(state))


class MultiLevelDelayedAcceptanceProposal(ProposalMethod):

    def __init__(self, coarseProposal, tgtMeasures, subchainLengths):

        # coarsest level uses MRW with coarseProposal as proposal
        self._proposalHierarchy = [MetropolisedRandomWalk(tgtMeasures[0],
                                                          coarseProposal)]

        for i in range(1, len(tgtMeasures)):

            self._proposalHierarchy.append(
                SurrogateTransitionProposal(tgtMeasures[i],
                                            self._proposalHierarchy[i - 1],
                                            subchainLengths[i]))

    def generate_proposal(self):
        return self._proposalHierarchy[-1].generate_proposal()


# role of client code
class MultiLevelDelayedAcceptance(MetropolisHastings):

    def __init__(self, proposalMethod, numLevels):
        pass

    def _acceptance_probability(self, proposal, state):
        pass
