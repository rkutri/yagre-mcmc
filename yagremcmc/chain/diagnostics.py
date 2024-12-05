import numpy as np

from collections import deque
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


class WelfordAccumulator(ChainDiagnostics):
    def __init__(self):
        self._dataSize = 0
        self._mean = None
        self._welfordM2 = None

    def mean(self):
        return self._mean

    def marginal_variance(self):
        """
        unbiased estimate of the marginal variance
        """
        if self._dataSize < 2:
            raise RuntimeError("Insufficient data for variance estimation.")
        return self._welfordM2 / (self._dataSize - 1)

    @property
    def nData(self):
        return self._dataSize

    def _update(self, state_vector):
        """
        Welford's algorithm for stable computation of variance.
        """
        if self._mean is None:
            self._mean = np.zeros_like(state_vector)
            self._welfordM2 = np.zeros_like(state_vector)

        delta = state_vector - self._mean

        # update mean
        self._dataSize += 1
        self._mean += delta / self._dataSize

        delta2 = state_vector - self._mean

        # update squared differences
        self._welfordM2 += delta * delta2

    def condition_number(self):

        margVar = self.marginal_variance()

        tol = 1e-12
        if np.min(margVar) < tol:
            raise RuntimeError("Singular marginal variance.")

        return np.max(margVar) / np.min(margVar)

    def process(self, transitionData):

        stateVector = transitionData.state.coefficient
        self._update(stateVector)

    def print_diagnostics(self, logger):

        logger.info(f"  - Estimated mean: {self.mean()}")
        logger.info(
            f"  - Estimated condition number: {self.condition_number():.4f}")

    def clear(self):
        self._dataSize = 0
        self._mean = None
        self._welfordM2 = None


class FullDiagnostics(ChainDiagnostics):

    def __init__(self):
        self._diagnostics = [AcceptanceRateDiagnostics(), WelfordAccumulator()]
        self._lag = None

    @property
    def lag(self):
        return self._lag

    @lag.setter
    def lag(self, lag):
        self._lag = lag
        self._diagnostics[0].lag = lag

    def global_acceptance_rate(self):
        return self._diagnostics[0].global_acceptance_rate()

    def marginal_variance(self):
        return self._diagnostics[1].marginal_variance()

    def mean(self):
        return self._diagnostics[1].mean()

    def process(self, transition_data):
        for diagnostic in self._diagnostics:
            diagnostic.process(transition_data)

    def print_diagnostics(self, logger):
        for diagnostic in self._diagnostics:
            diagnostic.print_diagnostics(logger)

    def clear(self):
        for diagnostic in self._diagnostics:
            diagnostic.clear()
