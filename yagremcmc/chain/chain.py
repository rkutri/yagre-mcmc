from numpy import array, sum

from yagremcmc.chain.diagnostics import ChainDiagnostics


class Chain:
    def __init__(self):

        self._trajectory = []
        self._diagnostics = ChainDiagnostics()

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def length(self):
        return len(self._trajectory)

    @property
    def diagnostics(self):
        return self._diagnostics

    def append(self, stateVector, isAccepted):

        self._trajectory.append(stateVector)

        if isAccepted:
            self._diagnostics.add_accepted()
        else:
            self._diagnostics.add_rejected()

    def accepted_states(self, startIdx):
        """
            startIdx: starting index of the trajectory, from which on to
                      consider the states.
        """

        states = self._trajectory[startIdx:]
        decisions = self._diagnostics.decisions[startIdx:]

        return [x for x, y in zip(states, decisions) if y == 1]

    def clear(self):

        self._trajectory = []

        self._diagnostics.clear()
