from abc import ABC, abstractmethod


class ProposalMethod(ABC):

    def __init__(self):

        super().__init__()
        self._state = None

    def get_state(self):
        return self._state

    def set_state(self, newState):
        self._state = newState

    @abstractmethod
    def generate_proposal(self):
        pass
