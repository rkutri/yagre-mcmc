from yagremcmc.parameter.interface import ParameterInterface


class EvaluationCache:

    def __init__(self, cacheSize: int) -> None:

        self._maxSize = cacheSize
        self._misses = 0

        self._keys = []
        self._cache = []

    @property
    def misses(self):
        return self._misses

    @property
    def keys(self):
        return self._keys

    def add(self, parameter: ParameterInterface, value) -> None:

        if len(self._cache) >= self._maxSize:

            self._keys.pop(0)
            self._cache.pop(0)

        self._keys.append(parameter)
        self._cache.append(value)

        return

    def contains(self, parameter: ParameterInterface) -> bool:

        if parameter in self._keys:
            return True

        else:

            self._misses += 1
            return False

    def __call__(self, parameter):

        paramIdx = self._keys.index(parameter)

        return self._cache[paramIdx]
