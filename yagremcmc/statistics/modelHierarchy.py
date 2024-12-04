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
            noiseModel: Hierarchy):

        BayesianModelHierarchyFactory.validate_model_components(
            [("data", data), ("prior", prior),
             ("fwdModel", fwdModel), ("noiseModel", noiseModel)]
        )

        self._nLevels = fwdModel.size
        self._data = data
        self._prior = prior
        self._fwdModel = fwdModel
        self._noiseModel = noiseModel

    def create_model(self):

        likelihoods = [
            AdditiveNoiseLikelihood(
                self._data.level(k),
                self._fwdModel.level(k),
                self._noiseModel.level(k))
            for k in range(self._nLevels)]

        modelHierarchy = [
            BayesianRegressionModel(
                self._prior.level(ell),
                likelihoods[ell])
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
