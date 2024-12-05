from yagremcmc.chain.diagnostics import WelfordAccumulator, FullDiagnostics
from yagremcmc.chain.target import UnnormalisedPosterior, ErrorCorrectionDecorator
from yagremcmc.chain.method.mlda import MLDA, MLDABuilder


class AdaptiveMLDA(MLDA):

    def __init__(
            self, tgtDensity, surrogateTargets, baseProposalCov, nSteps,
            idleSteps, nEstimation):

        targetDiagnostics = FullDiagnostics()
        surrogateDiagnostics = [WelfordAccumulator()
                                for _ in range(len(surrogateTargets))]

        super().__init__(tgtDensity, surrogateTargets, baseProposalCov,
                         nSteps, targetDiagnostics, surrogateDiagnostics)

        self._idleSteps = idleSteps
        self._adaptivityStart = idleSteps + nEstimation

    def _update_chain(self, stateVector):

        # reset accumulators after burn-in.
        if self._chain.length == self._idleSteps:

            for i in range(self.nSurrogates):
                self._proposalMethod.surrogate(i).diagnostics.clear()

            self._diagnostics.clear()

        # update error correction
        if self._chain.length > self._adaptivityStart:

            for i in range(self.nSurrogates):

                surrogate = self._proposalMethod.surrogate(i)

                if i < self.nSurrogates - 1:
                    tgtAcc = self._proposalMethod.surrogate(i + 1).diagnostics
                else:
                    tgtAcc = self.diagnostics

            surrogate.target.correction = surrogate.diagnostics.mean() - \
                tgtAcc.mean()

        super()._update_chain(stateVector)


class AdaptiveMLDABuilder(MLDABuilder):

    def __init__(self):

        print("Initialising Adaptive MLDA builder")
        super().__init__()

        self._idleSteps = None
        self._nEstimation = None

    @property
    def idleSteps(self):
        return self._idleSteps

    @idleSteps.setter
    def idleSteps(self, steps):
        self._idleSteps = steps

    @property
    def nEstimation(self):
        return self._nEstimation

    @nEstimation.setter
    def nEstimation(self, steps):
        self._nEstimation = steps

    def _validate_adaptivity_parameters(self):

        if self._idleSteps is None:
            raise ValueError("idle Steps not set.")
        if self._idleSteps < 0:
            raise ValueError("Invalid number of idle steps.")

        if self._nEstimation is None:
            raise ValueError("estimation steps not set.")
        if self._nEstimation < 2:
            raise ValueError("Invalid number of estimation steps.")

    def _validate_parameters(self):

        super()._validate_parameters()
        self._validate_adaptivity_parameters()

    def _create_adaptive_posteriors(self):
        print("creating adaptive posteriors")
        return [ErrorCorrectionDecorator(
                    UnnormalisedPosterior(self._bayesModel.level(k)))
                for k in range(self._bayesModel.size - 1)]

    def _create_adaptive_densities(self):
        print("creating adaptive densities")

        return [ErrorCorrectionDecorator(tgtDensity) for tgtDensity in self._surrTgts]

    def build_from_model(self):

        targetDensity = UnnormalisedPosterior(self._bayesModel.level(-1))
        surrogateDensities = self._create_adaptive_posteriors()

        return AdaptiveMLDA(targetDensity, surrogateDensities,
                            self._basePropCov, self._nSteps)

    def build_from_target(self):

        surrTgts = self._create_adaptive_densities()

        return AdaptiveMLDA(self._explicitTarget, surrTgts,
                            self._basePropCov, self._nSteps, self._idleSteps,
                            self._nEstimation)
