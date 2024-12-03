class TransitionData:

    REJECTED = 0
    ACCEPTED = 1

    def __init__(self, state, outcome):

        if not outcome in [TransitionData.REJECTED, TransitionData.ACCEPTED]:
            raise RuntimeError(f"invalid MC transition outcome: {outcome}")

        self._state = state
        self._outcome = outcome

    @property
    def state(self):
        return self._state

    @property
    def outcome(self):
        return self._outcome
