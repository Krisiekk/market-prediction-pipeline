import unittest

import numpy as np

from src.models.tune_h5_execution_ensemble import majority_vote


class H5ExecutionEnsembleTests(unittest.TestCase):
    def test_majority_vote_supports_three_and_five_voters(self):
        votes = np.array([
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
        ])
        np.testing.assert_array_equal(majority_vote(votes[:, :3]), [1, 0, 0])
        np.testing.assert_array_equal(majority_vote(votes), [1, 0, 0])

    def test_majority_vote_rejects_even_or_nonbinary_inputs(self):
        with self.assertRaises(ValueError):
            majority_vote([[1, 0]])
        with self.assertRaises(ValueError):
            majority_vote([[1, 2, 0]])


if __name__ == "__main__":
    unittest.main()
