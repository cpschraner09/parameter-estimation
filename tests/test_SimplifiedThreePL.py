import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        # Ensure the conditions sum to 500 trials
        condition1 = SignalDetection(55, 45, 10, 90)  # 55 correct responses
        condition2 = SignalDetection(60, 40, 15, 85)  # 60 correct responses
        condition3 = SignalDetection(75, 25, 20, 80)  # 75 correct responses
        condition4 = SignalDetection(90, 10, 25, 75)  # 90 correct responses
        condition5 = SignalDetection(95, 5, 30, 70)   # 95 correct responses

        # Initialize Experiment and add conditions
        self.experiment = Experiment()
        self.experiment.add_condition(condition1, label="Cond 1")
        self.experiment.add_condition(condition2, label="Cond 2")
        self.experiment.add_condition(condition3, label="Cond 3")
        self.experiment.add_condition(condition4, label="Cond 4")
        self.experiment.add_condition(condition5, label="Cond 5")

        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 375)
        self.assertEqual(summary["n_incorrect"], 125)
        self.assertEqual(summary["n_conditions"], 5)

    def test_predict_values(self):
        params = (1, 0)  # Arbitrary parameters
        predictions = self.model.predict(params)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

    def test_fit_model(self):
        self.model.fit()
        self.assertTrue(self.model._is_fitted)

    def test_get_parameters_before_fitting(self):
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    def test_get_parameters_after_fitting(self):
        self.model.fit()
        self.assertIsInstance(self.model.get_discrimination(), float)
        self.assertIsInstance(self.model.get_base_rate(), float)

if __name__ == "__main__":
    unittest.main()

