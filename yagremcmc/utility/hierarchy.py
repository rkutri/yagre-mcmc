from abc import ABC, abstractmethod
from typing import List


class Hierarchy(ABC):

    def __init__(self, nLevels):

        if nLevels < 1:
            raise ValueError(
                f"Trying to set up hierarchy with {nLevels} levels.")

        self._nLevels = nLevels

    @property
    def size(self):
        return self._nLevels

    def check_level_index(self, idx):

        if idx < 0 or idx == self._nLevels:
            raise ValueError(f"invalid level index. Trying to access level "
                             "{idx} in a hierarchy of {self._nLevels} levels.")

        else:
            return

    @property
    @abstractmethod
    def target(self):
        pass

    @abstractmethod
    def level(self, lvlIdx):
        pass


class shared(Hierarchy):

    def __init__(self, sharedComponent, nLevels: int):

        super().__init__(nLevels)

        self._sharedComponent = sharedComponent

    @property
    def target(self):
        return self._sharedComponent

    def level(self, lvlIdx):

        self.check_level_index(lvlIdx)

        return self._sharedComponent


class hierarchical(Hierarchy):

    def __init__(self, hierarchy: List):

        super().__init__(len(hierarchy))

        self._hierarchy = hierarchy

    @property
    def target(self):
        return self._hierarchy[-1]

    def level(self, lvlIdx):

        self.check_level_index(lvlIdx)
        return self._hierarchy[lvlIdx]
