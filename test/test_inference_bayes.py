import test.testSetup as setup

import numpy as np

from numpy.random import uniform
from inference.parameterLaw import IIDGaussian
from inference.noise import CentredGaussianIIDNoise
from inference.bayes import BayesianRegressionModel
from inference.metropolisedRandomWalk import MetropolisedRandomWalk
from inference.preconditionedCrankNicolson import PreconditionedCrankNicolson


def check_mean(means, trueParam):

    MTOL = 1e-1

    meanState = means[0]
    posteriorMean = means[1]

    np.allclose(posteriorMean.coefficient, trueParam.coefficient, atol=MTOL)
    np.allclose(meanState.coefficient, trueParam.coefficient, atol=2. * MTOL)


# define problem parameters
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))
alpha = 0.8
gamma = 0.4
T = 6.

# define forward map
fwdMap = setup.LotkaVolterraForwardMap(groundTruth, T, alpha, gamma)

# generate the data
dataSize = 5
inputData = [uniform(0.5, 1.5, 2) for _ in range(dataSize)]

dataNoiseVariance = 0.05
data = setup.generate_synthetic_data(fwdMap, inputData, dataNoiseVariance)

print("synthetic data generated")

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))
priorVariance = 1.
prior = IIDGaussian(priorMean, priorVariance)

# define a noise model
noiseVariance = dataNoiseVariance
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdMap, noiseModel)


def test_mrw():

    proposalVariance = 0.01
    mcmc = MetropolisedRandomWalk.from_bayes_model(statModel, proposalVariance)

    # run mcmc
    nSteps = 600
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain

    burnIn = 200
    thinningStep = 2

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


def test_pcn():

    stepSize = 0.001
    mcmc = PreconditionedCrankNicolson.from_bayes_model(statModel, stepSize)

    # run mcmc
    nSteps = 600
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain

    burnIn = 200
    thinningStep = 2

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)
