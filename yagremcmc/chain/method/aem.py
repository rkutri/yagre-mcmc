from yagremcmc.chain.method.mlda import MLDA, MLDABuilder
from yagremcmc.statistics.likelihood import AEMLikelihood
from yagremcmc.chain.transition import TransitionData


class AdaptiveErrorModel(MLDA):

    def __init__(self,
                 targetDensity,
                 surrogateDensities,
                 baseProposalCov,
                 nSteps,
                 targetDiagnostics,
                 surrogateDiagnostics):

        print("CONSTRUCTING ADAPTIVE ERROR MODEL MLDA")

        super().__init__(targetDensity,
                         surrogateDensities,
                         baseProposalCov,
                         nSteps,
                         targetDiagnostics,
                         surrogateDiagnostics)

    def _process_transition(self, transitionData):

        if transitionData.outcome == TransitionData.ACCEPTED:

            nSurrogates = super().nSurrogates

            for k in range(nSurrogates - 1):

                coarseSurrogate = super().surrogate(k)
                coarseModelEval = \
                    coarseSurrogate.target.likelihood.d_model_evaluation(
                        transitionData.proposal)

                fineSurrogate = super().surrogate(k + 1)
                fineModelEval = \
                    fineSurrogate.target.likelihood.query_model_evaluation(
                        transitionData.proposal)

                error = fineModelEval - coarseModelEval
                coarseSurrogate.target.likelihood.update_error_estimate(error)

            surrogate = super().surrogate(-1)

            surrogateModelEval = \
                surrogate.target.likelihood.query_model_evaluation(
                    transitionData.proposal)
            targetModelEval = \
                self._tgtDensity.likelihood.query_model_evaluation(
                    transitionData.proposal)

            error = targetModelEval - surrogateModelEval
            surrogate.target.likelihood.update_error_estimate(error)

        return super()._process_transition(transitionData)


class AEMBuilder(MLDABuilder):

    def __init__(self):

        super().__init__()

    def _validate_parameters(self):

        super()._validate_parameters()

        if self._bayesModel is None:
            raise NotImplementedError(
                "Adaptive error correction only makes "
                "sense if the target emerges from a Bayesian model.")

        for i in range(self._bayesModel.size):
            if not isinstance(self._bayesModel.level(
                    i).likelihood, AEMLikelihood):
                raise ValueError(f"Likelihood on level {i} is not adaptive.")

    def build_mlda(self, tgtPost, surPost, bpc, nS, tgtD, surD):
        return AdaptiveErrorModel(tgtPost, surPost, bpc, nS, tgtD, surD)
