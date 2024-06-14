import test.testSetup as setup

import numpy as np

from numpy.random import uniform
from inference.parameterLaw import IIDGaussian
from inference.noise import CentredGaussianIIDNoise
from inference.bayes import BayesianRegressionModel
from inference.preconditionedCrankNicolson import PreconditionedCrankNicolson


# define problem parameters
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))
alpha = 0.8
gamma = 0.4
T = 15.

# define forward map
fwdMap = setup.LotkaVolterraForwardMap(groundTruth, T, alpha, gamma)

# generate the data
dataSize = 4
inputData = [uniform(0.5, 1.5, 2) for _ in range(dataSize)]

dataNoiseVariance = 0.05
data = setup.generate_synthetic_data(fwdMap, inputData, dataNoiseVariance)

print("synthetic data generated")

# start with a prior centred around zero. Necessary for pCN
priorMean = setup.LotkaVolterraParameter(np.zeros(2))
priorVariance = 2.0
prior = IIDGaussian(priorMean, priorVariance)

# define a noise model
noiseVariance = dataNoiseVariance
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdMap, noiseModel)

# setup pcn
stepSize = 0.001
mcmc = PreconditionedCrankNicolson.from_bayes_model(statModel, stepSize)

# run mcmc
nSteps = 600
initState = setup.LotkaVolterraParameter(np.array([-0.6, -0.3]))
mcmc.run(nSteps, initState)

states = mcmc.chain

burnIn = 50
thinningStep = 2

mcmcSamples = states[burnIn::thinningStep]
meanState = setup.LotkaVolterraParameter(np.mean(states, axis=0))
posteriorMean = setup.LotkaVolterraParameter(np.mean(mcmcSamples, axis=0))


def test_mean():

    MTOL = 1e-1

    np.allclose(posteriorMean.coefficient, groundTruth.coefficient, atol=MTOL)
    np.allclose(meanState.coefficient, groundTruth.coefficient, atol=2. * MTOL)
