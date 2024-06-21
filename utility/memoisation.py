from parameter.interface import ParameterInterface


class EvaluationCache:

    def __init__(self, cacheSize: int) -> None:

        self.maxSize_ = cacheSize
        self.misses_ = 0

        self.keys_ = []
        self.cache_ = []

    def add(self, parameter: ParameterInterface, value) -> None:

        if (len(self.cache_) >= self.maxSize_):

            self.keys_.pop(0)
            self.cache_.pop(0)

        self.keys_.append(parameter)
        self.cache_.append(value)

        return


    def contains(self, parameter: ParameterInterface) -> bool:

        if parameter in self.keys_:
            return True

        else:

            self.misses_ += 1
            return False

    def __call__(self, parameter):

        paramIdx = self.keys_.index(parameter)

        return self.cache_[paramIdx]
