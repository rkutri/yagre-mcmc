class Chain:
    def __init__(self):

        self.stateVectors_ = []

    @property
    def trajectory(self):
        return self.stateVectors_

    @property
    def length(self):
        return len(self.stateVectors_)

    def append(self, stateVector):

        self.stateVectors_.append(stateVector)


# Thought for MLDA: Implement a hierarchical chain?
