import numpy as np


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    actual_fvc = (actual_fvc * (6399 - 827)) + 827
    predicted_fvc = (predicted_fvc * (6399 - 827)) + 827
    confidence *= 100
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)