from inference.interface import DataInterface


class InputOutputData(DataInterface):

    def __init__(self, inData, outData):

        assert len(inData) == len(outData)

        self.nData_ = len(inData)
        self.inData_ = inData
        self.outData_ = outData

    @property
    def size(self):

        return self.nData_

    @property
    def input(self):

        return self.inData_

    @property
    def output(self):

        return self.outData_
