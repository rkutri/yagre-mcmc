from abc import ABC, abstractmethod


class ChainDiagnostics:

    @abstractmethod
    def process(self, transitionData):
        pass

    @abstractmethod
    def print_diagnostics(self, logger):
        pass

    @abstractmethod
    def clear(self):
        pass
