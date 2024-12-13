from abc import ABC, abstractmethod
from typing import List


class HierarchyBase(ABC):

    def __init__(self, nLevels):

        if nLevels < 1:
            raise ValueError(
                f"Trying to set up hierarchy with {nLevels} levels.")

        self._nLevels = nLevels

    @property
    def size(self):
        return self._nLevels

    def validate_level_index(self, idx):

        if idx < -1 or self._nLevels <= idx:
            raise ValueError(f"invalid level index. Trying to access level "
                             "{idx} in a hierarchy of {self._nLevels} levels.")

        else:
            return

    @abstractmethod
    def level(self, lvlIdx):
        pass


class SharedComponent(HierarchyBase):

    def __init__(self, sharedComponent, nLevels: int):

        super().__init__(nLevels)

        self._sharedComponent = sharedComponent

    def level(self, lvlIdx):

        self.validate_level_index(lvlIdx)

        return self._sharedComponent


class Hierarchy(HierarchyBase):

    def __init__(self, hierarchy: List):

        super().__init__(len(hierarchy))

        self._hierarchy = hierarchy

    def level(self, lvlIdx):

        self.validate_level_index(lvlIdx)

        if lvlIdx == -1:
            return self._hierarchy[-1]
        else:
            return self._hierarchy[lvlIdx]
