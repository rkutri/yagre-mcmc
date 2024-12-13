from numpy import array


class Data:

    def __init__(self, dataArray):

        self.array_ = array(dataArray)

        self.size_ = dataArray.shape[0]
        self.dim_ = dataArray.shape[1]

    @property
    def size(self):
        return self.size_

    @property
    def dim(self):
        return self.dim_

    @property
    def array(self):
        return self.array_
