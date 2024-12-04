from yagremcmc.chain.method.mlda import MLDA, MLDABuilder
from yagremcmc.chain.target import UnnormalisedPosterior, TemperedUnnormalisedPosterior
from math import isclose


def validate_tempering_sequence(tSeq, nSurrogates):
    """
    Validates the tempering sequence for compatibility with the MLDA hierarchy.

    Args:
        tSeq (list[float]): Tempering sequence.
        nSurrogates (int): Number of surrogate levels in the hierarchy.

    Raises:
        ValueError: If the sequence length is mismatched or contains invalid values.
        RuntimeError: If the sequence does not target the true posterior.
    """
    if len(tSeq) != nSurrogates:
        raise ValueError(
            f"Tempering sequence length ({len(tSeq)}) does not "
            f"match the number of surrogates ({nSurrogates})."
        )

    for idx, gamma in enumerate(tSeq):
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(
                f"Invalid tempering parameter at index {idx}: {gamma} "
                "(must be in [0, 1])."
            )

    for i in range(1, nSurrogates):
        if tSeq[i - 1] > tSeq[i] and not isclose(tSeq[i - 1], tSeq[i], rel_tol=1e-9):
            raise ValueError(
                f"Non-monotonic tempering sequence at index {i}: "
                f"{tSeq[i - 1]} > {tSeq[i]}."
            )


class TemperedMLDA(MLDA):

    def __init__(self, targetDensity, surrogateDensities, basePropCov, nSteps, tSeq):

        super().__init__(targetDensity, surrogateDensities, basePropCov, nSteps)

        self._nSurrogates = self._proposalMethod.nSurrogates
        self._set_tempering(tSeq)

    def _set_tempering(self, tSeq):

        validate_tempering_sequence(tSeq, self._nSurrogates)
        self._tSeq = tSeq

        for i in range(self._nSurrogates):
            self._proposalMethod.target(i).tempering = self._tSeq[i]

    @property
    def temperingSequence(self):
        return self._tSeq

    @temperingSequence.setter
    def temperingSequence(self, tSeq):
        self._set_tempering(tSeq)


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

    def _create_tempered_posteriors(self):

        return [
            TemperedUnnormalisedPosterior(
                self._bayesModel.level(k), self._temperingSequence[k]
            )
            for k in range(self._bayesModel.size - 1)
        ]

    def build_from_model(self):
        """
        Builds a TemperedMLDA instance from the model hierarchy.

        Returns:
            TemperedMLDA: An MLDA chain with tempering.
        """
        super()._validate_parameters()
        validate_tempering_sequence(
            self._temperingSequence, self._bayesModel.size - 1
        )

        targetDensity = UnnormalisedPosterior(self._bayesModel.target)
        surrogateDensities = self._create_tempered_posteriors()

        return TemperedMLDA(
            targetDensity, surrogateDensities, self._basePropCov, self._nSteps, self._temperingSequence
        )

    def build_from_target(self):
        raise NotImplementedError(
            "Tempering only makes sense for target densities representing "
            "(unnormalised) posteriors."
        )
