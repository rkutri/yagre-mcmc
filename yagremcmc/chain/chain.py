from numpy import array, sum


class Chain:
    def __init__(self):

        self._trajectory = []

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def length(self):
        return len(self._trajectory)

    def append(self, stateVector):
        self._trajectory.append(stateVector)

    def clear(self):
        self._trajectory = []

