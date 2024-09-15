from abc import ABC, abstractmethod
from numpy.random import uniform
from yagremcmc.statistics.interface import DensityInterface
from yagremcmc.chain.interface import ProposalMethodInterface
from yagremcmc.chain.chain import Chain
from yagremcmc.chain.diagnostics import ChainDiagnostics


class UnnormalisedPosterior(DensityInterface):

    def __init__(self, model):

        self.model_ = model

    def evaluate_log(self, parameter):

        return self.model_.log_likelihood(parameter) \
            + self.model_.log_prior(parameter)


class MetropolisHastings(ABC):
    """
    Template class for Metropolis-Hastings-type chains
    """

    def __init__(self, targetDensity: DensityInterface,
                 proposalMethod: ProposalMethodInterface) -> None:

        self._tgtDensity = targetDensity
        self._proposalMethod = proposalMethod

        self._chain = Chain()
        self.diagnostics_ = ChainDiagnostics(self._chain)

    @property
    def chain(self):
        return self._chain

    @property
    def target(self):
        return self._tgtDensity

    @property
    def diagnostics(self):
        return self.diagnostics_


    @abstractmethod
    def _acceptance_probability(self, proposal, state):
        pass

    def _accept_reject(self, proposal, state):

        acceptProb = self._acceptance_probability(proposal, state)

        assert 0. <= acceptProb and acceptProb <= 1.

        decision = uniform(low=0., high=1., size=1)[0]

        if decision <= acceptProb:

            self.diagnostics_.add_accepted()
            return proposal

        else:
            self.diagnostics_.add_rejected()
            return state

    # TODO: switch to Python logging for verbosity
    def run(self, nSteps, initialState, verbose=True):

        self._chain.clear()

        state = initialState
        self._chain.append(state.coefficient)

        for n in range(nSteps - 1):

            if verbose:
                interval = nSteps // 20
                if (n % interval == 0):
                    if (n == 0):
                        print("Start Markov chain")
                    else:
                        ra = self.diagnostics_.rolling_acceptance_rate(
                            interval)
                        print(str(n) + " steps computed")
                        print("  - rolling acceptance rate: " + str(ra))

            self._proposalMethod.state = state
            proposal = self._proposalMethod.generate_proposal()

            state = self._accept_reject(proposal, state)

            self._chain.append(state.coefficient)

        return
