import unittest

import pytest

from learn.training_set import TrainingSet


class TestTrainingSet:

    @pytest.fixture
    def training_sets(self):
        return [
            TrainingSet([[1, 2, 3, 4], [5, 6, 7, 8]]),
            TrainingSet([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        ]

    def test_examples(self, training_sets):
        assert [2, 3] == [training_set.examples for training_set in training_sets]

    def test_features(self, training_sets):
        assert [3, 3] == [training_set.features for training_set in training_sets]

    def test_targets(self, training_sets):
        assert [[4, 8], [4, 8, 12]] == [list(training_set.targets) for training_set in training_sets]

    def test_invalid_input_type(self):
        with pytest.raises(ValueError) as e:
            TrainingSet('stuff')
        assert e.exconly() == 'ValueError: data must be a list, found str'

        with pytest.raises(ValueError) as e:
            TrainingSet(2)
        assert e.exconly() == 'ValueError: data must be a list, found int'

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError) as e:
            TrainingSet([1, 2, 3])
        assert e.exconly() == 'ValueError: data must be 2-dimensional, found 1 dimensions'

        with pytest.raises(ValueError) as e:
            TrainingSet([[[1]], [[2]], [[3]]])
        assert e.exconly() == 'ValueError: data must be 2-dimensional, found 3 dimensions'

    def test_invalid_columns(self):
        with pytest.raises(ValueError) as e:
            TrainingSet([[1], [2], [3]])
        assert e.exconly() == 'ValueError: data must have at least 2 colums, found 1'

        with pytest.raises(ValueError) as e:
            TrainingSet([[], []])
        assert e.exconly() == 'ValueError: data must have at least 2 colums, found 0'
