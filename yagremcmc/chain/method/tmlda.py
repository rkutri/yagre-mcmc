from yagremcmc.chain.method.mlda import MLDA, MLDABuilder
from yagerecmc.chain.target import UnnormalisedPosterior, TemperedUnnormalisedPosterior


def validate_tempering_sequence(tSeq, nSurrogates):
    """
    Raises:
        ValueError: If the sequence length is mismatched or contains
                    invalid values.
        RuntimeError: If the sequence does not target the true posterior.
    """

    if len(tSeq) != nSurrogates:
        raise ValueError(
            f"Tempering sequence length ({len(tSeq)}) does not "
            "match the number of surrogates ({nSurrogates})."
        )

    for idx, gamma in enumerate(tSeq):
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(
                f"Invalid tempering parameter at index {idx}: {gamma} "
                "(must be in [0, 1]).")

    for i in range(1, nSurrogates):
        if tSeq[i - 1] > tSeq[i]:
            raise ValueError(
                f"Non-monotonic tempering sequence at index {i}: "
                "{tSeq[i-1]} > {tSeq[i]}.")

    return


class TemperedMLDA(MLDA):

    def __init__(self, targetDensity, surrogateDensities, basePropCov, nSteps):
        super().__init__(targetDensity, surrogateDensities, basePropCov, nSteps)

    def set_tempering_sequence(self, tSeq):

        nSurrogates = self._proposalMethod.depth

        validate_tempering_sequence(tSeq, nSurrogates)

        for i in range(nSurrogates):
            self._proposalMethod.get_surrogate(i).target.tempering = tSeq[i]


class TemperedMLDABuilder(MLDABuilder):

    def __init__(self):

        super().__init__()
        self._temperingSequence = None

    @property
    def temperingSequence(self):
        return self._temperingSequence

    @temperingSequence.setter
    def temperingSequence(self, tSeq):
        self._temperingSequence = tSeq

    def build_from_model(self):

        super()._validate_parameters()
        validate_tempering_sequence(
            self._temperingSequence,
            self._bayesModel.size - 1)

        targetDensity = UnnormalisedPosterior(self._bayesModel.target)

        surrogateDensities = []

        for k in range(self._bayesModel.size - 1):
            surrogateDensitites.append(
                TemperedUnnormalisedPosterior(
                    self._bayesModel.level(k), self._temperingSequence[k]))

        return MLDA(targetDensity, sorrogateDensities,
                    self._basePropCov, self._nSteps)

    def build_from_target(self):

        raise NotImplementedError("Tempering only makes sense for target "
                                  "densities representing (unnormalised) posteriors.")
