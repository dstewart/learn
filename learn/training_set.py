import numpy as np


class TrainingSet:
    
    def __init__(self, data):
        if type(data) != list:
            raise ValueError('data must be a list, found {}'.format(type(data).__name__))

        self._data = np.array(data)

        dimensions = len(self._data.shape)
        if  dimensions != 2:
            raise ValueError('data must be 2-dimensional, found {} dimensions'.format(dimensions))
        
        columns = self._data.shape[1]
        if columns < 2:
            raise ValueError('data must have at least 2 colums, found {}'.format(columns))

    @property
    def examples(self):
        return self._data.shape[0]
