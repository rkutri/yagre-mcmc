from numpy import array, average


class ChainDiagnostics:

    def __init__(self) -> None:

        self._decisions = []

    @property
    def decisions(self):
        return self._decisions

    def add_accepted(self):
        self._decisions.append(1)

    def add_rejected(self):
        self._decisions.append(0)

    def rolling_acceptance_rate(self, lag):
        return average(array(self._decisions[-lag:]))

    def global_acceptance_rate(self):
        return average(array(self._decisions))

    def clear(self):
        self._decisions = []
