from numpy import exp

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.chain.target import UnnormalisedPosterior


class SurrogateTransitionProposal(ProposalMethod, MetropolisHastings):

    def __init__(self, targetDensity, proposalMethod, subChainLength):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of surrogate transiton needs"
                             " to be derived from MetropolisHastings")

        MetropolisHastings.__init__(self, targetDensity, proposalMethod)
        ProposalMethod.__init__(self)

        self._subChainLength = subChainLength

    def generate_proposal(self):

        if self.state_ is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        # the proposal method of surrogate transition is always itself
        # derived from a Metropolis-Hastings algorithm
        self._proposalMethod.run(self._subChainLength, self._state)

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

    def __init__(self, coarseProposalCov, surrTgtMeasures, subChainLengths):
        """
        surrTgtMeasures: DensityInterface

            Measures to be used as targets for the surrogate transitions. Does
            not include the final target measure of the entire chain.
        """

        self._nSurrogates = len(surrTgtMeasures)

        assert len(subChainLengths) == self._nSurrogates

        # lowest level is just a MRW
        self._baseSurrogateChain = MetropolisedRandomWalk(
            surrTgtMeasures[0], coarseProposalCov)
        self._baseSubChainLength = subChainLengths[0]

        self._proposalHierarchy = []

        if self._nSurrogates > 1:

            # proposal for the next level is the MRW on the lowest level
            self._proposalHierarchy.append(
                SurrogateTransitionProposal(
                    surrTgtMeasures[1],
                    self._baseSurrogateChain,
                    subChainLengths[1]))

        # the remaining levels consist of surrogate transition proposals,
        # where each one uses the next coarser one as its own proposal
        for i in range(2, self._nSurrogates):
            self._proposalHierarchy.append(
                SurrogateTransitionProposal(surrTgtMeasures[i],
                                            self._proposalHierarchy[i - 2],
                                            subChainLengths[i]))

    @property
    def finestSurrogate(self):

        if self._nSurrogates > 1:
            return self._proposalHierarchy[-1]
        else:
            return self._baseSurrogateChain

    def generate_proposal(self):

        # if the only proposal is the coarse chain
        if len(self._proposalHierarchy) > 0:
            return self._proposalHierarchy[-1].generate_proposal()

        else:

            self._baseSurrogateChain.run(self._baseSubChainLength, self._state, verbose=False)

            return self._stateType(
                self._baseSurrogateChain.chain.trajectory[-1])


class MLDA(MetropolisHastings):

    def __init__(self, targetDensity, surrogateDensities,
                 coarseProposalCov, subChainLengths):

        self._finestTarget = surrogateDensities[-1]

        surrogateTgts = surrogateDensities
        proposal = MLDAProposal(
            coarseProposalCov,
            surrogateDensities,
            subChainLengths)

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

        self._coarsePropCov = None
        self._scLengths = None
        self._surrTgts = None

    @property
    def coarseProposalCovariance(self):
        return self._coarsePropCov

    @coarseProposalCovariance.setter
    def coarseProposalCovariance(self, cov):
        self._coarsePropCov = cov

    @property
    def subChainLengths(self):
        return self._scLengths

    @subChainLengths.setter
    def subChainLengths(self, scLengths):
        self._scLengths = scLengths

    @property
    def surrogateTargets(self):
        return self._surrTgts

    @surrogateTargets.setter
    def surrogateTargets(self, tgts):
        self._surrTgts = tgts

    def _validate_parameters(self):

        if self._coarsePropCov is None:
            raise ValueError("Coarse proposal covariance not set for MLDA")

        if self._scLengths is None:
            raise ValueError("Subchain lengths not set for MLDA")

        if self._surrTgts is None:
            raise ValueError("Surrogate targets not set for MLDA")

        if not len(self._scLengths) == len(self._surrTgts):
            raise ValueError(
                "Number of sub-chain lengths does not match number of surrogate targets")

        return

    def build_from_model(self):

        self._validate_parameters()

        targetDensity = UnnormalisedPosterior(self._bayesModel)

        return MLDA(targetDensity, self._surrTgts,
                    self.coarsePropCov, self._scLengths)

    def build_from_target(self):

        self._validate_parameters()

        return MLDA(self._explicitTarget, self._surrTgts,
                    self._coarsePropCov, self._scLengths)
