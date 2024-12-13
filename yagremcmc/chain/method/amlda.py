from yagremcmc.chain.method.mlda import MLDA, MLDABuilder
from yagremcmc.statistics.likelihood import AdaptiveErrorCorrection
from yagremcmc.chain.transition import TransitionData


class AdaptiveErrorCorrectionMLDA(MLDA):

    def _process_transition(self, transitionData):

        super()._process_transition(transitionData)

        if transitionData.outcome == TransitionData.ACCEPTED:

            nSurrogates = super().nSurrogates

            for k in range(nSurrogates - 1):

                coarseSurrogate = super().surrogate(k)
                coarseModelEval = \
                    coarseSurrogate.target.likelihood.cached_model_evaluation(
                        transitionData.proposal)

                fineSurrogate = super().surrogate(k + 1)
                fineModelEval = \
                    fineSurrogate.target.likelihood.cached_model_evaluation(
                        transitionData.proposal)

                error = fineModelEval - coarseModelEval
                coarseSurrogate.target.likelihood.update_error_estimate(error)

            surrogate = super().surrogate(-1)

            surrogateModelEval = \
                surrogate.target.likelihood.cached_model_evaluation(
                    transitionData.proposal)
            targetModelEval = \
                self._targetDensity.likelihood.cached_model_evaluation(
                    transitionData.proposal)

            error = targetModelEval - surrogateModelEval
            surrogate.target.likelihood.update_error_estimate(error)


class AdaptiveErrorCorrectionMLDABuilder(MLDABuilder):

    def __init__(self):

        super().__init__(self)

    def _validate_parameters(self):

        super()._validate_parameters()

        if self._bayesModel is None:
            raise NotImplementedError(
                "Adaptive error correction only makes "
                "sense if the target emerges from a Bayesian model.")

        for i in range(self._bayesModel.size):
            if not isinstance(self._bayesModel.level(
                    i).likelihood, AdaptiveErrorCorrection):
                raise ValueError(f"Likelihood on level {i} is not adaptive.")
