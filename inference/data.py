class Data:
    """
    data_ : list of numpy.ndarray
    """

    def __init__(self, data):

        self.data_ = data

    @property
    def size(self):
        return len(self.data_)

    @property
    def data(self):
        return self.data_
