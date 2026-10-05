import unittest

import numpy as np
import pandas as pd

from src.models.evaluate_h5_arx import _combined_vote


class H5ExecutionARXTests(unittest.TestCase):
    def test_sixth_equal_weight_vote_only_breaks_or_preserves_existing_majority(self):
        frame = pd.DataFrame({
            "LogisticRegression": [1, 1, 0, 0],
            "ExtraTreesClassifier": [1, 1, 0, 0],
            "SVC": [1, 1, 0, 1],
            "RandomForestClassifier": [0, 0, 1, 1],
            "HistGradientBoostingClassifier": [0, 0, 1, 1],
            "arx_prediction": [0, 1, 0, 1],
            "Ensemble_HardVote_5": [1, 1, 0, 1],
        })
        up_count, six_vote, covered = _combined_vote(frame)
        base_vote = frame.Ensemble_HardVote_5.to_numpy()
        self.assertEqual(up_count[0], 3)  # tie -> abstain
        self.assertTrue(np.isnan(six_vote[0]))
        self.assertFalse(covered[0])
        np.testing.assert_array_equal(six_vote[covered].astype(int), base_vote[covered])


if __name__ == "__main__":
    unittest.main()
