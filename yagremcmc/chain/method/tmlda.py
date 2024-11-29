from yagremcmc.chain.method.mlda import MLDA, MLDABuilder
from yagerecmc.chain.target import UnnormalisedPosterior, TemperedUnnormalisedPosterior


class TemperedMLDA(MLDA):
    pass


class TemperedMLDABuilder(MLDABuilder):

    def __init__(self):

        super().__init__()
        self._temperingSequence = None


    def validate_tempering_sequence(self, temperingSeq: List[float]):
        """
        Validates the tempering sequence.

        Raises:
            ValueError: If the sequence length is mismatched or contains
                        invalid values.
            RuntimeError: If the sequence does not target the true posterior.
        """
	
        # target posterior is not tempered
        nSurrogates = self._bayesModel.size - 1

        if len(temperingSeq) != nSurrogates:
            raise ValueError(
                f"Tempering sequence length ({len(temperingSeq)}) does not "
                "match the number of surrogates ({nSurrogates})."
            )

        for idx, gamma in enumerate(temperingSeq):
            if gamma < 0.0 or gamma > 1.0:
                raise ValueError(
                    f"Invalid tempering parameter at index {idx}: {gamma} "
                    "(must be in [0, 1]).")

        for i in range(1, nSurrogates):
            if temperingSeq[i - 1] > temperingSeq[i]:
                raise ValueError(
                    f"Non-monotonic tempering sequence at index {i}: "
                    "{temperingSeq[i-1]} > {temperingSeq[i]}.")

    @property
    def temperingSequence(self):
        return self._temperingSequence

    @temperingSequence.setter
    def temperingSequence(self, tSeq):

        self.validate_tempering_sequence(tSeq)
        self._temperingSequence = tSeq

    def build_from_model(self):

        super()._validate_parameters()
        self.validate_tempering_sequence(self._temperingSequence)

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
