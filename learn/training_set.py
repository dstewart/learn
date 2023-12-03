import numpy as np


class TrainingSet:
    
    def __init__(self, data: list):
        if type(data) != list:
            raise ValueError('data must be a list, found {}'.format(type(data).__name__))

        data = np.array(data)

        dimensions = len(data.shape)
        if  dimensions != 2:
            raise ValueError('data must be 2-dimensional, found {} dimensions'.format(dimensions))
        
        columns = data.shape[1]
        if columns < 2:
            raise ValueError('data must have at least 2 colums, found {}'.format(columns))

        self._features = data[:, 0:-1]
        self._targets = data[:, -1]

    @property
    def examples(self) -> int:
        return self._features.shape[0]
    
    @property
    def features(self) -> int:
        return self._features.shape[1]
    
    @property
    def targets(self) -> np.ndarray:
        return self._targets
