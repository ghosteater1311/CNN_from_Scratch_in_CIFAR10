import numpy as np

def cross_entropy_loss(predictions, targets):
    clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
    n = len(predictions)
    result = -(np.sum(targets * np.log(clipped)) / n)
    return result