

class DataSet(object):

    def __init__(self, train: tuple, validation: tuple, test: tuple) -> None:
        assert len(train) == 2, 'Expected train data to be a tuple of 2: (X, y)'
        assert len(validation) == 2, 'Expected train data to be a tuple of 2: (X, y)'
        assert len(test) == 2, 'Expected train data to be a tuple of 2: (X, y)'
        self.train = train
        self.validation = validation
        self.test = test

    def train_set(self):
        return self.train

    def validation_set(self):
        return self.train

    def test_set(self):
        return self.test
