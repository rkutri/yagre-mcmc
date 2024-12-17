from abc import ABC, abstractmethod
from yagremcmc.parameter.interface import ParameterInterface
from yagremcmc.model.evaluation import AEMEvaluation


class Cache(ABC):
    """
    Template class for caching mechanisms.
    """

    def __init__(self, cacheSize: int) -> None:
        """
        Initialize the cache with a maximum size.

        Parameters:
            cacheSize (int): Maximum number of elements the cache can hold.
        """
        self._maxSize = cacheSize

        self._misses = 0
        self._hits = 0

        self._keys = []

    @property
    def misses(self):
        return self._misses

    @property
    def hits(self):
        return self._hits

    def contains(self, parameter: ParameterInterface) -> bool:
        return parameter in self._keys

    @abstractmethod
    def add(self, parameter: ParameterInterface, cacheValue) -> None:
        pass

    @abstractmethod
    def retrieve(self, parameter: ParameterInterface):
        pass


class EvaluationCache(Cache):

    def __init__(self, cacheSize: int) -> None:

        super().__init__(cacheSize)
        self._cache = []

    def _evict_oldest(self):

        self._keys.pop(0)
        self._cache.pop(0)

    def add(self, parameter: ParameterInterface, cacheValue) -> None:

        if len(self._cache) >= self._maxSize:
            self._evict_oldest()

        self._keys.append(parameter)
        self._cache.append(cacheValue)

    def retrieve(self, parameter: ParameterInterface):

        if self.contains(parameter):
            self._hits += 1
            paramIdx = self._keys.index(parameter)
            return self._cache[paramIdx]

        self._misses += 1
        raise RuntimeError("Evaluation Cache missed.")


class AEMCache(Cache):
    """
    Cache specifically designed to store AEM-relevant evaluations.
    """

    # Large cache can become memory-costly quickly, as we store evaluations of
    # the forward model.
    CACHESIZE = 3

    def __init__(self) -> None:

        super().__init__(AEMCache.CACHESIZE)

        self._fmCache = []  # Cache for forward model evaluations
        self._llCache = []  # Cache for log-likelihood evaluations

    def _evict_oldest(self):

        self._keys.pop(0)
        self._fmCache.pop(0)
        self._llCache.pop(0)

    def _move_to_back(self, index):
        """Move the cached element at 'index' to the back of all caches."""

        self._keys.append(self._keys.pop(index))
        self._fmCache.append(self._fmCache.pop(index))
        self._llCache.append(self._llCache.pop(index))

    def add(self, parameter: ParameterInterface,
            cacheValue: AEMEvaluation) -> None:

        if not isinstance(cacheValue, AEMEvaluation):
            raise ValueError("AEM Cache is supposed to store AEM-relevant "
                             "evaluations.")

        if len(self._keys) >= self._maxSize:
            self._evict_oldest()

        self._keys.append(parameter)
        self._fmCache.append(cacheValue.forwardModelEvaluation)
        self._llCache.append(cacheValue.logLikelihoodEvaluation)

    def retrieve(self, parameter: ParameterInterface):
        """
        Retrieve the forward model and log-likelihood evaluations for a
        parameter.

        Parameters:
            parameter (ParameterInterface): The parameter whose evaluations to
                                            retrieve.

        Returns:
            tuple: A tuple containing
                   (forwardModelEvaluation, logLikelihoodEvaluation).

        Raises:
            RuntimeError: If the parameter is not in the cache.
        """
        if self.contains(parameter):
            self._hits += 1
            paramIdx = self._keys.index(parameter)

            # Move the retrieved item to the back of the cache
            self._move_to_back(paramIdx)

            fmEval = self._fmCache[-1]
            llEval = self._llCache[-1]

            return fmEval, llEval

        self._misses += 1
        raise RuntimeError("Retrieving evaluation for parameter failed "
                           "(cache miss).")
