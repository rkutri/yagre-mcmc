import numpy as np

from yagremcmc.chain.interface import ChainDiagnostics
from yagremcmc.chain.transition import TransitionData
from yagremcmc.statistics.estimation import WelfordAccumulator


class DummyDiagnostics(ChainDiagnostics):
    def process(self, transitionData):
        return

    def print_diagnostics(self, logger):
        return

    def clear(self):
        return


class AcceptanceRateDiagnostics(ChainDiagnostics):
    def __init__(self):

        self._decisions = []
        self._lag = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, value):

        if value <= 0:
            raise ValueError("Lag must be a positive integer.")
        self._lag = value

    def rolling_acceptance_rate(self):
        if not self._lag:
            raise RuntimeError("Lag for rolling acceptance rate not set.")
        if len(self._decisions) < self._lag:
            raise RuntimeError(
                "Insufficient data for rolling acceptance rate.")
        return np.sum(self._decisions[-self._lag:]) / self._lag

    def global_acceptance_rate(self):
        return np.sum(self._decisions) / \
            len(self._decisions) if self._decisions else 0.0

    def process(self, transition_data):
        if transition_data.outcome == TransitionData.REJECTED:
            self._decisions.append(0)
        elif transition_data.outcome == TransitionData.ACCEPTED:
            self._decisions.append(1)
        else:
            raise ValueError("Invalid acceptance decision.")

    def print_diagnostics(self, logger):
        try:
            rAccept = self.rolling_acceptance_rate()
            logger.info(f"  - Rolling acceptance rate: {rAccept:.4f}")
        except RuntimeError as e:
            logger.warning(f"  - Rolling acceptance rate unavailable: {e}")

    def clear(self):
        self._decisions = []


class FullDiagnostics(ChainDiagnostics):

    def __init__(self):
        self._diagnostics = AcceptanceRateDiagnostics()
        self._accumulator = WelfordAccumulator()
        self._lag = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, lag):
        self._lag = lag

    def global_acceptance_rate(self):
        return self._diagnostics.global_acceptance_rate()

    def marginal_variance(self):
        return self._accumulator.marginal_variance()

    def mean(self):
        return self._accumulator.mean()

    def process(self, transitionData):

        self._diagnostics.process(transitionData)
        self._accumulator.update(transitionData.state.coefficient)

    def print_diagnostics(self, logger):

        self._diagnostics.lag = self._lag
        self._diagnostics.print_diagnostics(logger)

        logger.info(f"  - Estimated mean: {self._accumulator.mean()}")
        logger.info("  - Estimated condition number: "
                    f"{self._accumulator.condition_number():.4f}")

    def clear(self):

        self._diagnostics.clear()
        self._accumulator.reset()
