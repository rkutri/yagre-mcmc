# TODO: compute and store diagnostics: accept-rate
class Chain:
    def __init__(self):

        self.stateVectors_ = []

    @property
    def trajectory(self):
        return self.stateVectors_

    def add_state_vector(self, vector):

        self.stateVectors_.append(vector)
