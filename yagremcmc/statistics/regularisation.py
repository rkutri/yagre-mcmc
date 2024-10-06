import numpy as np


def regularised_marginal_variance_weights(margVar):

    dim = margVar.size

    minEV = np.min(margVar)
    maxEV = np.max(margVar)

    if np.abs(minEV) < 1.e-8:
        raise ValueError("Singular Marginal Variance")

    regIntensity = np.min([-0.01 + 0.01 * maxEV / minEV, 1.])

    regScaling = (1. - regIntensity) * margVar + regIntensity * np.ones(dim)
    maxScale = np.max(regScaling)

    return regScaling / maxScale
