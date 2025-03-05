import numpy as np
import scipy.optimize as opt
from scipy.special import expit  # Sigmoid function (inverse logit)
from src.Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize the model with an Experiment object."""
        if not isinstance(experiment, Experiment):
            raise TypeError("Expected an Experiment instance.")
        
        self.experiment = experiment
        self._discrimination = None
        self._logit_base_rate = None
        self._is_fitted = False
        self._difficulties = np.array([2, 1, 0, -1, -2])  # Fixed difficulty parameters
        self.n_correct = sum(cond.hits for cond in self.experiment.conditions)
        self.n_incorrect = sum(cond.misses for cond in self.experiment.conditions)
        print(f"DEBUG: Initialized n_correct = {self.n_correct}, n_incorrect = {self.n_incorrect}")

    def summary(self):
        return {
            "n_total": self.n_correct + self.n_incorrect,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_conditions": len(self.experiment.conditions)
        }
    
    def predict(self, parameters):
        """Compute the probability of a correct response for each condition."""
        a, logit_c = parameters
        c = expit(logit_c)  # Convert logit_c to probability
        
        probabilities = c + (1 - c) / (1 + np.exp(-a * (0 - self._difficulties)))
        return probabilities
    
    def negative_log_likelihood(self, parameters):
        """Compute the negative log-likelihood of the data."""
        probabilities = self.predict(parameters)
        
        n_correct = np.array([sdt.n_correct_responses() for sdt in self.experiment.conditions])
        n_total = np.array([sdt.n_total_responses() for sdt in self.experiment.conditions])
        n_incorrect = n_total - n_correct
        
        log_likelihood = (
            np.sum(n_correct * np.log(probabilities)) +
            np.sum(n_incorrect * np.log(1 - probabilities))
        )
        
        return -log_likelihood  # Minimize this value

    def fit(self):
        """Find the best-fitting discrimination and base rate parameters."""
        initial_guess = [1.0, 0.0]  # Start with a reasonable guess for (a, logit_c)
        result = opt.minimize(self.negative_log_likelihood, initial_guess, method="L-BFGS-B")
        
        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed to converge.")
    
    def get_discrimination(self):
        """Return the estimated discrimination parameter."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        """Return the estimated base rate parameter."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return expit(self._logit_base_rate)  # Convert logit to probability

