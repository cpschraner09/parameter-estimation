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

    def test_fit_stability_and_prediction_alignment(self):  #Integration test

        # Create conditions with exactly 100 signal trials (hits + misses = 100)
        cond1 = SignalDetection(55, 45, 0, 1)
        cond2 = SignalDetection(60, 40, 0, 1)
        cond3 = SignalDetection(75, 25, 0, 1)
        cond4 = SignalDetection(90, 10, 0, 1)
        cond5 = SignalDetection(95, 5, 0, 1)

        exp = Experiment()
        exp.add_condition(cond1, label="Cond 1")
        exp.add_condition(cond2, label="Cond 2")
        exp.add_condition(cond3, label="Cond 3")
        exp.add_condition(cond4, label="Cond 4")
        exp.add_condition(cond5, label="Cond 5")
        
        model = SimplifiedThreePL(exp)
        
        # Fit the model several times and collect parameters and predictions.
        params_list = []
        predictions_list = []
        for _ in range(3):
            model.fit()
            params_list.append((model._discrimination, model._logit_base_rate))
            predictions_list.append(model.predict((model._discrimination, model._logit_base_rate)))
        
        tol = 1e-4
        # Check that parameter estimates are stable.
        for i in range(1, len(params_list)):
            self.assertAlmostEqual(params_list[0][0], params_list[i][0], delta=tol)
            self.assertAlmostEqual(params_list[0][1], params_list[i][1], delta=tol)
        
        # Expected accuracy rates for signal trials.
        observed = [0.55, 0.60, 0.75, 0.90, 0.95]
        for pred, obs in zip(predictions_list[0], observed):
            self.assertAlmostEqual(pred, obs, delta=0.05)


    def test_signal_detection_corruption(self):    #Corruption test
 
        # Non-integer inputs.
        with self.assertRaises(TypeError):
            SignalDetection(55.5, 45, 1, 99)
        with self.assertRaises(TypeError):
            SignalDetection(55, 45.0, 1, 99)
        
        # Negative values.
        with self.assertRaises(ValueError):
            SignalDetection(-1, 46, 1, 99)
        
        # Zero signal trials (hits + misses == 0).
        with self.assertRaises(ValueError):
            SignalDetection(0, 0, 1, 99)
        
        # Zero noise trials (falseAlarms + correctRejections == 0).
        with self.assertRaises(ValueError):
            SignalDetection(55, 45, 0, 0)

if __name__ == "__main__":
    unittest.main()

