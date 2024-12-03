import numpy as np

from yagremcmc.chain.interface import ChainDiagnostics
from yagremcmc.chain.transition import TransitionData


class DummyDiagnostics(ChainDiagnostics):

    def process(self, transitionData):
        return

    def print_diagnostics(self, logger):
        return

    def clear(self):
        return


class AcceptanceRateDiagnostics(ChainDiagnostics):

    def __init__(self) -> None:

        self._decisions = []
        self._lag = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, lag):
        self._lag = lag

    @property
    def decisions(self):
        return self._decisions

    def rolling_acceptance_rate(self):

        if self._lag is None:
            raise RuntimeError("lag of rolling acceptance rate not set")

        if len(self._decisions) < self._lag:
            raise RuntimeError("Insufficient data for rolling acceptance rate")

        return np.mean(np.array(self._decisions[-self._lag:]))

    def global_acceptance_rate(self):
        return np.mean(np.array(self._decisions))

    def process(self, transitionData):

        if transitionData.outcome == TransitionData.REJECTED:
            self._decisions.append(0)
        elif transitionData.outcome == TransitionData.ACCEPTED:
            self._decisions.append(1)
        else:
            raise RuntimeError("Invalid acceptance decision")

    def print_diagnostics(self, logger):

        rAccept = self.rolling_acceptance_rate()
        logger.info(f"  - rolling acceptance rate: {rAccept}")

    def clear(self):
        self._decisions = []
