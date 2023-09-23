class Dataset:
    """Class for managing training, validation, and test datasets."""

    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test
