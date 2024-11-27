from typing import List, Tuple
from yagremcmc.utility.hierarchy import Hierarchy, shared, hierarchical
from yagremcmc.statistics.likelihood import AdditiveNoiseLikelihood
from yagremcmc.statistics.bayesModel import BayesianRegressionModel


class BayesianModelHierarchyFactory:
    """
    Factory class for constructing a Bayesian model hierarchy.
    """

    INVALID_TYPE_MSG = ("Argument '{}' must be derived from the Hierarchy "
                        "base class. Received type: {}.")
    SIZE_MISMATCH_MSG = ("Hierarchies have mismatched sizes. The following "
                         "mismatches were found:\n{}")

    def __init__(
        self,
        data: Hierarchy,
        prior: Hierarchy,
        fwdModel: Hierarchy,
        noiseModel: Hierarchy,
        temperingSeq: List[float] = None,
    ):
        """
        Initializes the factory with hierarchical model components.

        Args:
            data: Hierarchy object representing the observed data.
            prior: Hierarchy object for prior distributions.
            fwdModel: Hierarchy object for forward models.
            noiseModel: Hierarchy object for noise models.
            temperingSeq: Optional list of tempering parameters.
                          Must be monotonically increasing and end with 1.0.

        Raises:
            ValueError: If hierarchy sizes are inconsistent or tempering
                        sequence is invalid.
        """
        self.validate_model_components(
            [("data", data), ("prior", prior),
             ("fwdModel", fwdModel), ("noiseModel", noiseModel)]
        )

        self._nLevels = fwdModel.size
        self._data = data
        self._prior = prior
        self._fwdModel = fwdModel
        self._noiseModel = noiseModel

        self.validate_tempering(temperingSeq)
        self._temperingSeq = temperingSeq

    def create_model(self):
        """
        Constructs the hierarchical Bayesian regression model.

        Returns:
            A hierarchical Bayesian model composed of BayesianRegressionModel
            instances.
        """
        likelihoods = [
            AdditiveNoiseLikelihood(
                self._data.level(k),
                self._fwdModel.level(k),
                self._noiseModel.level(k),
                self._temperingSeq[k] if self._temperingSeq else 1.0,
            )
            for k in range(self._nLevels)
        ]

        modelHierarchy = [
            BayesianRegressionModel(self._prior.level(ell), likelihoods[ell])
            for ell in range(self._nLevels)
        ]

        return hierarchical(modelHierarchy)

    @classmethod
    def validate_model_components(
            cls, hierarchies: List[Tuple[str, Hierarchy]]):
        """
        Validates that all provided hierarchies are instances of `Hierarchy`
        and have the same size.

        Args:
            hierarchies: A list of (name, Hierarchy) pairs to validate.

        Raises:
            ValueError: If any hierarchy is not a `Hierarchy` instance or
                        if sizes are inconsistent.
        """
        sizeDict = {name: instance.size for name, instance in hierarchies}

        if len(set(sizeDict.values())) > 1:
            report = "\n".join(
                f" - {name}: size {size}" for name,
                size in sizeDict.items())
            raise ValueError(cls.SIZE_MISMATCH_MSG.format(report))

    def validate_tempering(self, temperingSeq: List[float]):
        """
        Validates the tempering sequence.

        Raises:
            ValueError: If the sequence length is mismatched or contains
                        invalid values.
            RuntimeError: If the sequence does not target the true posterior.
        """
        if temperingSeq is None:
            return

        if len(temperingSeq) != self._nLevels:
            raise ValueError(
                f"Tempering sequence length ({len(temperingSeq)}) does not "
                "match the number of levels ({self._nLevels})."
            )

        if temperingSeq[-1] != 1.0:
            raise RuntimeError(
                "Tempering sequence does not target the true posterior "
                "(last value must be 1.0).")

        for idx, gamma in enumerate(temperingSeq):
            if gamma < 0.0 or gamma > 1.0:
                raise ValueError(
                    f"Invalid tempering parameter at index {idx}: {gamma} "
                    "(must be in [0, 1]).")

        for i in range(1, self._nLevels):
            if temperingSeq[i - 1] > temperingSeq[i]:
                raise ValueError(
                    f"Non-monotonic tempering sequence at index {i}: "
                    "{temperingSeq[i-1]} > {temperingSeq[i]}.")
