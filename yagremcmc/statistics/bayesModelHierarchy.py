from typing import List, Tuple
from yagremcmc.utility.hierarchy import Hierarchy
from yagremcmc.statistics.bayesModel import BayesianRegressionModel


class BayesianRegressionModelHierarchy(Hierarchy):

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
    ):
        self._validate_input([
            ("data", data),
            ("prior", prior),
            ("fwdModel", fwdModel),
            ("noiseModel", noiseModel)
        ])

        # after validation, all hierarchies are assured to have the same size
        nLevels = fwdModel.size

        super().__init__(nLevels)

        self._modelHierarchy = [
            BayesianRegressionModel(
                data.level(k),
                prior.level(k),
                fwdModel.level(k),
                noiseModel.level(k))
            for k in range(nLevels)
        ]

    def level(self, lvlIdx):

        self.check_level_index(lvlIdx)

        return self._modelHierarchy[lvlIdx]

    @classmethod
    def _validate_input(cls, hierarchies: List[Tuple[str, Hierarchy]]):

        for name, instance in hierarchies:
            if not isinstance(instance, Hierarchy):
                raise ValueError(cls.INVALID_TYPE_MSG.format(
                    name, type(instance).__name__))

        sizeDict = {name: instance.size for name, instance in hierarchies}

        uniqueSizes = set(size_dict.values())

        if len(unique_sizes) > 1:

            report = "\n".join(f" -  {name}: size {size}"
                               for name, size in size_dict.items())

            raise ValueError(cls.SIZE_MISMATCH_MSG.format(report))
