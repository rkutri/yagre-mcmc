from numpy import exp

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.chain.diagnostics import DummyDiagnostics, AcceptanceRateDiagnostics
from yagremcmc.utility.hierarchy import Hierarchy


class SurrogateTransitionProposal(MetropolisHastings, ProposalMethod):

    def __init__(self, targetDensity, proposalMethod, nSteps, diagnostics):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of surrogate transiton needs"
                             " to be derived from MetropolisHastings")

        super().__init__(targetDensity, proposalMethod, diagnostics)
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

    def __init__(self, surrTgtMeasures, nSteps,
                 baseProposalCov, diagnosticsList):
        """
        surrTgtMeasures: DensityInterface

            Measures to be used as targets for the surrogate transitions. Does
            not include the final target measure of the entire chain.
        """

        self._nSurrogates = len(surrTgtMeasures)

        assert len(nSteps) > 0 and len(nSteps) == self._nSurrogates

        self._surrogateHierarchy = None
        self._baseSurrogate = self._build_base_surrogate(
            surrTgtMeasures[0], nSteps[0], baseProposalCov, diagnosticsList[0])

        if self._nSurrogates > 1:
            self._surrogateHierarchy = self._build_surrogate_hierarchy(
                surrTgtMeasures, nSteps, diagnosticsList)

    def target(self, tIdx):

        if tIdx < 0 or tIdx >= self._nSurrogates:
            raise ValueError(f"invalid target index: {tIdx}")

        if tIdx == 0:
            return self._baseSurrogate.target
        else:
            return self._surrogateHierarchy[sIdx - 1].target

    @property
    def depth(self):
        return self._nSurrogates

    def generate_proposal(self):

        # base chain is the only surrogate, and thus surrogateHierarchy
        # is empty
        if self._surrogateHierarchy is None:

            self._baseSurrogate.run(
                self._baseChainLength, self._state, verbose=False)

            return self._stateType(self._baseSurrogate.chain.trajectory[-1])

        else:

            self._surrogateHierarchy[-1].set_state(self._state)
            return self._surrogateHierarchy[-1].generate_proposal()

    def _build_base_surrogate(self, tgtDensity, nSteps, propCov, baseDiagnostics):

        # as the chain includes the initial state, the total length is the
        # number of steps + 1
        self._baseChainLength = nSteps + 1

        return MetropolisedRandomWalk(tgtDensity, propCov, baseDiagnostics)

    def _build_surrogate_hierarchy(
            self, surrTgtMeasures, nSteps, diagnosticsList):

        hierarchy = []

        # proposal for the next level is the MRW on the lowest level
        hierarchy.append(
            SurrogateTransitionProposal(
                surrTgtMeasures[1], self._baseSurrogate, nSteps[1], diagnosticsList[1]))

        # the remaining levels consist of surrogate transition proposals,
        # where each one uses the next baser one as its own proposal
        for i in range(2, self._nSurrogates):
            hierarchy.append(
                SurrogateTransitionProposal(surrTgtMeasures[i],
                                            hierarchy[i - 2],
                                            nSteps[i],
                                            diagnosticsList[i]))

        return hierarchy


class MLDA(MetropolisHastings):

    def __init__(self, targetDensity, surrogateDensities,
                 baseProposalCov, nSteps, targetDiagnostics, surrogateDiagnosticsList):

        self._finestTarget = surrogateDensities[-1]

        proposal = MLDAProposal(
            surrogateDensities,
            nSteps,
            baseProposalCov,
            surrogateDiagnosticsList)

        super().__init__(targetDensity, proposal, targetDiagnostics)

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
        self._surrDgnstList = None
        self._tgtDgnst = None

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

    @property
    def surrogateDiagnostics(self):
        return self._surrDgnstList

    @surrogateDiagnostics.setter
    def surrogateDiagnostics(self, surrDgnstList):
        self._surrDgnstList = surrDgnstList

    def _validate_parameters(self):

        if self._bayesModel is not None:

            if not isinstance(self._bayesModel, Hierarchy):
                raise ValueError("MLDA requires a hierarchy of models.")

            if self._surrTgts is not None:
                raise ValueError("Cannot set explicit surrogate targets for "
                                 "a hierarchy of Bayesian models.")

            if not len(self._nSteps) == self._bayesModel.size - 1:
                raise ValueError("Number of sub-chain lengths does not match "
                                 "the size of the model hierarchy.")

            if self._surrDgnstList is not None:
                if not len(self._surrDgnstList) == self._bayesModel.size - 1:
                    raise ValueError("Number of diagnostics does not match "
                                     "the size of the model hierarchy")

        if self._explicitTarget is not None:

            if self._surrTgts is None:
                raise ValueError("Surrogate targets not set for MLDA")

            if not len(self._nSteps) == len(self._surrTgts):
                raise ValueError(
                    "Number of sub-chain lengths does not match number of "
                    "surrogate targets")

            if self._surrDgnstList is not None:
                if not len(self._surrDgnstList) == len(self._surrTgts):
                    raise ValueError("Number of diagnostics does not match "
                                     "the size of the model hierarchy")

        if self._basePropCov is None:
            raise ValueError("Coarse proposal covariance not set for MLDA")

        if self._nSteps is None:
            raise ValueError("Subchain lengths not set for MLDA")

        return

    def create_diagnostics(self, nSurrogates):

        if self._tgtDgnst is None:
            self._tgtDgnst = AcceptanceRateDiagnostics()

        if self._surrDgnstList is None:
            self._surrDgnstList = [DummyDiagnostics()
                                   for _ in range(nSurrogates)]

        return

    def build_from_model(self):

        targetDensity = UnnormalisedPosterior(self._bayesModel.target)

        surrogateDensities = []

        nSurrogates = self._bayesModel.size - 1

        for k in range(nSurrogates):
            surrogateDensities.append(
                UnnormalisedPosterior(self._bayesModel.level(k)))

        self.create_diagnostics(nSurrogates)

        print(f"target: {self._tgtDgnst}")

        return MLDA(targetDensity,
                    surrogateDensities,
                    self._basePropCov,
                    self._nSteps,
                    self._tgtDgnst,
                    self._surrDgnstList
                    )

    def build_from_target(self):

        self.create_diagnostics(len(self._surrTgts))

        return MLDA(self._explicitTarget, self._surrTgts,
                    self._basePropCov, self._nSteps,
                    self._tgtDgnst, self._surrDgnstList)
