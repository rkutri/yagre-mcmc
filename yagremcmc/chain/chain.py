# TODO: compute and store diagnostics: accept-rate
class Chain:
    def __init__(self):

        self.stateVectors_ = []
        self.nAccept_ = 0

    @property
    def trajectory(self):
        return self.stateVectors_

    def add_state_vector(self, vector, isAccepted):

        self.stateVectors_.append(vector)

        if isAccepted:
            self.nAccept_ += 1

    def acceptance_rate(self):

        # initial state does not count as accepted
        return self.nAccept_ / (len(self.stateVectors_) - 1)

    
