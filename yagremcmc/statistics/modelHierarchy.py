from typing import List, Tuple
from yagremcmc.utility.hierarchy import Hierarchy
from yagremcmc.statistics.likelihood import (AdditiveGaussianNoiseLikelihood,
					     AdaptiveErrorCorrection)
from yagremcmc.statistics.bayesModel import BayesianRegressionModel


class BayesianModelHierarchy(Hierarchy):

    INVALID_TYPE_MSG = ("Argument '{}' must be derived from the Hierarchy "
                        "base class. Received type: {}.")
    SIZE_MISMATCH_MSG = ("Hierarchies have mismatched sizes. The following "
                         "mismatches were found:\n{}")

    def __init__(self, likelihood: Hierarchy, prior: Hierarchy):

        BayesianModelHierarchy.validate_model_components([
            ("prior", prior),
            ("likelihood", likelihood)
        ])

        modelHierarchy = [
            BayesianRegressionModel(likelihood.level(ell), prior.level(ell))
                for ell in range(likelihood.size)]

        super().__init__(modelHierarchy)

    @classmethod
    def validate_model_components(cls,
                                  hierarchies: List[Tuple[str, Hierarchy]]):
        """
        Validates that all provided hierarchies are instances of `Hierarchy`
        and have the same size.

        Args:
            hierarchies: A list of (name, Hierarchy) pairs to validate.

        Raises:
            ValueError: If any hierarchy is not a `Hierarchy` instance or if 
                        sizes are inconsistent.
        """
        sizeDict = {name: instance.size for name, instance in hierarchies}

        # Validate all are Hierarchy instances
        for name, instance in hierarchies:
            if not isinstance(instance, Hierarchy):
                raise ValueError(
		    cls.INVALID_TYPE_MSG.format(name, type(instance).__name__))

        # Validate consistent sizes
        if len(set(sizeDict.values())) > 1:
            report = "\n".join(
                f" - {name}: size {size}" for name, size in sizeDict.items()
            )
            raise ValueError(cls.SIZE_MISMATCH_MSG.format(report))

