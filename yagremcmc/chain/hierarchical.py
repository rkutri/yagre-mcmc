from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.metropolisHastings import MetropolisHastings

class HierarchicalProposal(ProposalMethodInterface):

    @property
    def state(self):
        pass

    @state.setter
    def state(self, newState):
        pass

    def generate_proposal(self):
        pass


# role of client code
class HierarchicalMetropolisHastings(MetropolisHastings):

    def __init__(self, proposalMethod, numLevels):
        pass

    def _acceptance_probability(self, proposal, state):
        pass
