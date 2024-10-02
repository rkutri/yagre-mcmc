from yagremcmc.chain.diagnostics import ChainDiagnostics


class Chain:
    def __init__(self):

        self.stateVectors_ = []
        self._diagnostics = ChainDiagnostics()

    @property
    def trajectory(self):
        return self.stateVectors_

    @property
    def length(self):
        return len(self.stateVectors_)

    @property
    def diagnostics(self):
        return self._diagnostics

    def append(self, stateVector, isAccepted):

        self.stateVectors_.append(stateVector)

        if isAccepted:
            self._diagnostics.add_accepted()
        else:
            self._diagnostics.add_rejected()

    def clear(self):

        self.stateVectors_ = []

        self._diagnostics.clear()
