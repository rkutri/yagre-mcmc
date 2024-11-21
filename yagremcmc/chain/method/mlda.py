from numpy import exp

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.chain.target import UnnormalisedPosterior


class SurrogateTransitionProposal(MetropolisHastings, ProposalMethod):

    def __init__(self, targetDensity, proposalMethod, nSteps):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of surrogate transiton needs"
                             " to be derived from MetropolisHastings")

        super().__init__(targetDensity, proposalMethod)

        self._nSteps = nSteps

    def generate_proposal(self):

        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        # the proposal method of surrogate transition is always itself
        # derived from a Metropolis-Hastings algorithm
        self._proposalMethod.run(self._nSteps + 1, self._state, verbose=False)

        return self._stateType(self._proposalMethod.chain.trajectory[-1])

    def _acceptance_probability(self, proposal, state):

        ratio = exp(
            self._tgtDensity.evaluate_log(proposal) +
            self._proposalMethod.target.evaluate_log(state) -
            self._proposalMethod.target.evaluate_log(proposal) -
            self._tgtDensity.evaluate_log(state))

        return ratio if ratio < 1. else 1.


class MLDAProposal(ProposalMethod):
    """
    Represents a composite proposal and is purely a ProposalMethod.
    """

    def __init__(self, surrTgtMeasures, nSteps, baseProposalCov):
        """
        surrTgtMeasures: DensityInterface

            Measures to be used as targets for the surrogate transitions. Does
            not include the final target measure of the entire chain.
        """

        self._nSurrogates = len(surrTgtMeasures)

        assert len(nSteps) > 0 and len(nSteps) == self._nSurrogates

        self._surrogateHierarchy = None
        self._baseSurrogate = self._build_base_surrogate(
            surrTgtMeasures[0], nSteps[0], baseProposalCov)

        if self._nSurrogates > 1:
            self._surrogateHierarchy = self._build_surrogate_hierarchy(
                surrTgtMeasures, nSteps)

    def generate_proposal(self):

        L = len(self._surrogateHierarchy)

        # base chain is the only surrogate, and thus surrogateHierarchy
        # is empty
        if L == 0:

            self._baseSurrogate.run(
                self._baseChainLength, self._state, verbose=False)

            return self._stateType(self._baseSurrogate.chain.trajectory[-1])

        else:

            self._surrogateHierarchy[-1].set_state(self._state)
            return self._surrogateHierarchy[-1].generate_proposal()

    def _build_base_surrogate(self, tgtDensity, nSteps, propCov):

        # as the chain includes the initial state, the total length is the
        # number of steps + 1
        self._baseChainLength = nSteps + 1

        return MetropolisedRandomWalk(tgtDensity, propCov)

    def _build_surrogate_hierarchy(self, surrTgtMeasures, nSteps):

        hierarchy = []

        # proposal for the next level is the MRW on the lowest level
        hierarchy.append(
            SurrogateTransitionProposal(
                surrTgtMeasures[1], self._baseSurrogate, nSteps[1]))

        # the remaining levels consist of surrogate transition proposals,
        # where each one uses the next baser one as its own proposal
        for i in range(2, self._nSurrogates):
            hierarchy.append(
                SurrogateTransitionProposal(surrTgtMeasures[i],
                                            self._surrogateHierarchy[i - 2],
                                            nSteps[i]))

        return hierarchy


class MLDA(MetropolisHastings):

    def __init__(self, targetDensity, surrogateDensities,
                 baseProposalCov, nSteps):

        self._finestTarget = surrogateDensities[-1]

        proposal = MLDAProposal(surrogateDensities, nSteps, baseProposalCov)

        super().__init__(targetDensity, proposal)

    def _acceptance_probability(self, proposal, state):

        ratio = exp(
            self._tgtDensity.evaluate_log(proposal) +
            self._finestTarget.evaluate_log(state) -
            self._finestTarget.evaluate_log(proposal) -
            self._tgtDensity.evaluate_log(state))

        return ratio if ratio < 1. else 1.


class MLDABuilder(ChainBuilder):

    def __init__(self):

        super().__init__()

        self._basePropCov = None
        self._nSteps = None
        self._surrTgts = None

    @property
    def baseProposalCovariance(self):
        return self._basePropCov

    @baseProposalCovariance.setter
    def baseProposalCovariance(self, cov):
        self._basePropCov = cov

    @property
    def subChainLengths(self):
        return self._nSteps

    @subChainLengths.setter
    def subChainLengths(self, nSteps):
        self._nSteps = nSteps

    @property
    def surrogateTargets(self):
        return self._surrTgts

    @surrogateTargets.setter
    def surrogateTargets(self, tgts):
        self._surrTgts = tgts

    def _validate_parameters(self):

        if self._basePropCov is None:
            raise ValueError("Coarse proposal covariance not set for MLDA")

        if self._nSteps is None:
            raise ValueError("Subchain lengths not set for MLDA")

        if self._surrTgts is None:
            raise ValueError("Surrogate targets not set for MLDA")

        if not len(self._nSteps) == len(self._surrTgts):
            raise ValueError(
                "Number of sub-chain lengths does not match number of surrogate targets")

        return

    def build_from_model(self):

        self._validate_parameters()

        targetDensity = UnnormalisedPosterior(self._bayesModel)

        return MLDA(targetDensity, self._surrTgts,
                    self.basePropCov, self._nSteps)

    def build_from_target(self):

        self._validate_parameters()

        return MLDA(self._explicitTarget, self._surrTgts,
                    self._basePropCov, self._nSteps)
