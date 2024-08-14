from numpy import array, average


class ChainDiagnostics:

    def __init__(self, chain) -> None:

        self.decisions_ = []

    def add_accepted(self):
        self.decisions_.append(1)

    def add_rejected(self):
        self.decisions_.append(0)

    def rolling_acceptance_rate(self, lag):
        return average(array(self.decisions_[-lag:]))

    def global_acceptance_rate(self):
        return average(array(self.decisions_))
