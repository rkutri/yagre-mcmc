from abc import ABC, abstractmethod


class ProposalMethodInterface(ABC):

    @property
    @abstractmethod
    def state(self):
        pass

    @state.setter
    def state(self, newState):
        pass

    @abstractmethod
    def generate_proposal(self):
        pass
