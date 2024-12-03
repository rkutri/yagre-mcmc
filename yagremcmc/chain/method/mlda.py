from numpy import exp

from yagremcmc.chain.proposal import ProposalMethod
from yagremcmc.chain.metropolisHastings import MetropolisHastings
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.builder import ChainBuilder
from yagremcmc.chain.target import UnnormalisedPosterior
from yagremcmc.chain.diagnostics import DummyDiagnostics, AcceptanceRateDiagnostics
from yagremcmc.utility.hierarchy import Hierarchy


class SurrogateTransitionProposal(MetropolisHastings, ProposalMethod):

    def __init__(self, targetDensity, proposalMethod, diagnostics, nSteps):

        if not isinstance(proposalMethod, MetropolisHastings):
            raise ValueError("Proposal method of surrogate transiton needs"
                             " to be derived from MetropolisHastings")

        super().__init__(targetDensity, proposalMethod, diagnostics)
        self._chainLength = nSteps + 1

    def generate_proposal(self):

        if self._state is None:
            raise ValueError(
                "Trying to generate proposal with undefined state")

        # the proposal method of surrogate transition is always itself derived
        # from a Metropolis-Hastings algorithm, so it provides a run() method
        self._proposalMethod.run(self._chainLength, self._state, verbose=False)

        return self._stateType(self._proposalMethod.chain.trajectory[-1])

    def _acceptance_probability(self, proposal, state):

        ratio = exp(
            self._tgtDensity.evaluate_log(proposal) +
            self._proposalMethod.target.evaluate_log(state) -
            self._proposalMethod.target.evaluate_log(proposal) -
            self._tgtDensity.evaluate_log(state))

        return ratio if ratio < 1. else 1.


class SurrogateChainHierarchy(Hierarchy):

    def __init__(self, tgtMeasures, diagnosticsList, basePropCov, nSteps):

        nLevels = len(tgtMeasures)

        if nLevels == 0:
            raise ValueError("Surrogate hierarchy must contain at least one "
                             "surrogate.")

        if not len(diagnosticsList) == nLevels:
            raise ValueError("Mismatch in number of surrogate targets and "
                             "corresponding chain diagnostics")

        super().__init__(nLevels)

        self._hierarchy = [
            MetropolisedRandomWalk(
                tgtMeasures[0],
                basePropCov,
                diagnosticsList[0])]

        for level in range(1, nLevels):
            self._hierarchy.append(SurrogateTransitionProposal(
                tgtMeasures[level],
                self._hierarchy[level - 1],
                diagnosticsList[level],
                nSteps[level]))

    @property
    def highest(self):
        return self._hierarchy[-1]

    def level(self, lvlIdx):

        self.validate_level_index(lvlIdx)
        return self._hierarchy[lvlIdx]


class MLDAProposal(ProposalMethod):
    """
    Represents a composite proposal and is purely a ProposalMethod.
    """

    def __init__(self, surrTgtMeasures, surrogateDiagnostics,
                 baseProposalCov, nSteps):

        self._surrogateHierarchy = SurrogateChainHierarchy(
            surrTgtMeasures, surrogateDiagnostics, baseProposalCov, nSteps)

        self._baseChainLength = nSteps[0] + 1

    def target(self, tIdx):

        if tIdx < 0 or tIdx >= self._surrogateHierarchy.size:
            raise IndexError(f"invalid target index: {tIdx}")

        return self._surrogateHierarchy.level(tIdx).target

    @property
    def depth(self):
        return self._surrogateHierarchy.size

    def generate_proposal(self):

        if self.depth == 1:

            surrogate = self._surrogateHierarchy.level(0)

            surrogate.run(
                self._baseChainLength,
                self._state,
                verbose=False)
            return self._stateType(surrogate.chain.trajectory[-1])

        else:

            surrogate = self._surrogateHierarchy.highest
            surrogate.set_state(self._state)

            return surrogate.generate_proposal()


class MLDA(MetropolisHastings):

    def __init__(self, targetDensity, surrogateDensities,
                 baseProposalCov, nSteps, targetDiagnostics, surrogateDiagnosticsList):

        self._finestTarget = surrogateDensities[-1]

        proposal = MLDAProposal(
                surrogateDensities,
                surrogateDiagnosticsList,
                baseProposalCov,
                nSteps
            )

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
