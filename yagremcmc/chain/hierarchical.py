from numpy import exp

from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.metropolisedRandomWalk import MetropolisedRandomWalk


class SurrogateTransition(MetropolisHastings):

    def __init__(self, targetDensity, proposalMethod):
        super.__init__(targetDensity, proposalMethod)

    def _acceptance_probability(self, proposal, state):

        # the proposalMethod of a SurrogateTransition is always derived from
        # a MetropolisHastings class
        densityRatio = exp(self._targetDensity.evaluate_log(proposal)
                           - self._targetDensity.evaluate_log(state)
                           + self._proposalMethod.target.evaluate_log(state)
                           - self._proposalMethod.target.evaluate_log(proposal))


class CompositeProposal(MetropolisHastings, ProposalMethodInterface):

    def __init__(self, targetDensity, proposalMethod, subchainLength):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of a composite proposal needs"
                             " to be derived from MetropolisHastings class")

        # TODO: CompositeProposal is its own MetropolisHastings! no need
        #       for an extra member. I just need to figure out the correct
        #       super() call to set up the Metropolis Hastings
        self._surrogateTransition = SurrogateTransition(
            targetDensity, proposalMethod)
        self._subchainLength = subchainLength

        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, newState):
        self._state = newState

    def generate_proposal(self):

        if self.state_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        self._proposalMethod.state = self._state
        self._surrogateTransition.run(self._subchainLength, self._state)

        return self._st.trajectory[-1]


class HierarchicalProposal(ProposalMethodInterface):

    def __init__(self, coarseProposal, tgtMeasures):

        self._state = None
        self._depth = len(tgtMeasures)

        # start with coarsest level
        self._proposalHierarchy = [MetropolisedRandomWalk(tgtMeasures[0],
                                                          coarseProposal)]

        for i in range(1, self._depth):

            self._proposalHierarchy.append(
                CompositeProposal(tgtMeasures[i],
                                  self._proposalHierarchy[i-1], child.target))

    @ property
    def state(self):
        return self._state

    @ state.setter
    def state(self, newState):
        self._state = newState

    @ property
    def depth(self):
        return self._depth

    def generate_proposal(self):

        self._proposalHierarchy[-1].state = self._state

        return self._proposalHierarchy[-1].generate_proposal()


# role of client code
class HierarchicalMetropolisHastings(MetropolisHastings):

    def __init__(self, proposalMethod, numLevels):
        pass

    def _acceptance_probability(self, proposal, state):
        pass
