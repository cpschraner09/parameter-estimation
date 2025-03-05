import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        """Set up an experiment with five conditions and predefined correct/incorrect responses."""
        self.conditions = [
            SignalDetection(55, 45, 10, 90),  # 55% correct
            SignalDetection(60, 40, 15, 85),  # 60% correct
            SignalDetection(75, 25, 20, 80),  # 75% correct
            SignalDetection(90, 10, 25, 75),  # 90% correct
            SignalDetection(95, 5, 30, 70)    # 95% correct
        ]
        self.experiment = Experiment()
        for idx, cond in enumerate(self.conditions):
            self.experiment.add_condition(cond, label=f"Condition {idx+1}")
        
        self.model = SimplifiedThreePL(self.experiment)

    # ===== Initialization Tests =====
    def test_valid_initialization(self):
        """Test that the model initializes properly with a valid Experiment object."""
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 375)
        self.assertEqual(summary["n_incorrect"], 125)
        self.assertEqual(summary["n_conditions"], 5)

    def test_invalid_initialization(self):
        """Test that constructor raises an error when given an invalid input."""
        with self.assertRaises(TypeError):
            SimplifiedThreePL("invalid_input")  # Should be an Experiment object

    # ===== Prediction Tests =====
    def test_prediction_values_within_bounds(self):
        """Test that predict() outputs values between 0 and 1."""
        params = (1.0, 0.0)  # Arbitrary parameter values
        predictions = self.model.predict(params)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

    def test_higher_base_rate_increases_probability(self):
        """Test that increasing the base rate increases predicted probabilities."""
        params_low = (1.0, -2.0)  # Lower base rate (logit_c = -2)
        params_high = (1.0, 2.0)   # Higher base rate (logit_c = 2)
        predictions_low = self.model.predict(params_low)
        predictions_high = self.model.predict(params_high)
        self.assertTrue(np.all(predictions_high > predictions_low))

    def test_higher_difficulty_reduces_probability_when_a_is_positive(self):
        """Test that higher difficulty values result in lower probabilities when a is positive."""
        params = (1.0, 0.0)  # Discrimination is positive
        predictions = self.model.predict(params)
        self.assertTrue(np.all(np.diff(predictions) > 0))  # Probabilities should increase as difficulty decreases

    def test_higher_ability_increases_probability_when_a_is_positive(self):
        """Test that higher ability parameters increase probability when a is positive."""
        params_low = (0.5, 0.0)  # Lower discrimination
        params_high = (2.0, 0.0) # Higher discrimination
        predictions_low = self.model.predict(params_low)
        predictions_high = self.model.predict(params_high)
        self.assertTrue(np.all(predictions_high > predictions_low))

    def test_prediction_with_known_parameters(self):
        """Test that predict() produces expected values given fixed parameters."""
        params = (1.0, 0.0)  # Fixed parameters
        expected_probabilities = np.array([0.119, 0.269, 0.5, 0.731, 0.881])
        predictions = self.model.predict(params)
        np.testing.assert_allclose(predictions, expected_probabilities, atol=0.1)

    # ===== Parameter Estimation Tests =====
    def test_negative_log_likelihood_improves_after_fitting(self):
        """Test that fitting the model reduces negative log-likelihood."""
        initial_params = [1.0, 0.0]
        initial_nll = self.model.negative_log_likelihood(initial_params)
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model.get_base_rate()])
        self.assertLess(fitted_nll, initial_nll)

    def test_higher_discrimination_when_data_is_steeper(self):
        """Test that a larger estimate of 'a' is returned for steeper accuracy curves."""
        steep_conditions = [
            SignalDetection(20, 80, 5, 95),  # Low accuracy
            SignalDetection(40, 60, 10, 90),
            SignalDetection(60, 40, 15, 85),
            SignalDetection(80, 20, 20, 80),  # High accuracy
            SignalDetection(95, 5, 25, 75)
        ]
        steep_experiment = Experiment()
        for cond in steep_conditions:
            steep_experiment.add_condition(cond)
        steep_model = SimplifiedThreePL(steep_experiment)
        steep_model.fit()
        self.model.fit()
        self.assertGreater(steep_model.get_discrimination(), self.model.get_discrimination())

    def test_parameter_access_before_fitting(self):
        """Ensure the user cannot access parameters before fitting."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    # ===== Integration Tests =====
    def test_model_convergence(self):
        """Test that model parameters remain stable after multiple fits."""
        self.model.fit()
        param1 = self.model.get_discrimination()
        param2 = self.model.get_base_rate()
        self.model.fit()  # Refit should not change parameters
        param3 = self.model.get_discrimination()
        param4 = self.model.get_base_rate()
        self.assertAlmostEqual(param1, param3, places=2)
        self.assertAlmostEqual(param2, param4, places=2)

    def test_model_prediction_alignment(self):
        """Ensure that model predictions align with known accuracy rates after fitting."""
        self.model.fit()
        predictions = self.model.predict([self.model.get_discrimination(), self.model.get_base_rate()])
        expected_accuracies = np.array([0.55, 0.60, 0.75, 0.90, 0.95])
        np.testing.assert_allclose(predictions, expected_accuracies, atol=0.1)

    # ===== Corruption Tests =====
    def test_private_attribute_protection(self):
        """Ensure that users cannot modify private attributes directly."""
        with self.assertRaises(AttributeError):
            self.model._discrimination = 5.0  # Should not be allowed
        with self.assertRaises(AttributeError):
            self.model._logit_base_rate = 2.0  # Should not be allowed
        with self.assertRaises(AttributeError):
            self.model._is_fitted = True  # Should not be allowed

if __name__ == "__main__":
    unittest.main()
