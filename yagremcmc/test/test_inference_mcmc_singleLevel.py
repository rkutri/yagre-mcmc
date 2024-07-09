import pytest
import yagremcmc.test.testSetup as setup

import numpy as np

from numpy.random import uniform, seed
from yagremcmc.inference.parameterLaw import IIDGaussian
from yagremcmc.inference.noise import CentredGaussianIIDNoise
from yagremcmc.inference.bayesModel import BayesianRegressionModel
from yagremcmc.inference.metropolisedRandomWalk import MetropolisedRandomWalk
from yagremcmc.inference.preconditionedCrankNicolson import PreconditionedCrankNicolson
from yagremcmc.model.forwardModel import ForwardModel


def check_mean(means, trueParam):

    MTOL = 1e-2

    meanState = means[0]
    posteriorMean = means[1]

    np.allclose(posteriorMean.coefficient, trueParam.coefficient, atol=MTOL)
    np.allclose(meanState.coefficient, trueParam.coefficient, atol=2. * MTOL)


config = {'T': 6., 'alpha': 0.8, 'gamma': 0.4, 'nData': 5, 'dataDim': 2}
design = np.array([
    np.array([0.1, 0.9]),
    np.array([0.5, 0.5]),
    np.array([1., 0.5]),
    np.array([0.5, 1.]),
    np.array([2.5, 1.5])
])

# define forward problem
solver = setup.LotkaVolterraSolver(design, config)
fwdModel = ForwardModel(solver)

# define problem parameters
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))

dataNoiseVariance = 0.05
data = setup.generate_synthetic_data(groundTruth, solver, dataNoiseVariance)

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))
priorVariance = 1.
prior = IIDGaussian(priorMean, priorVariance)

# define a noise model
noiseVariance = dataNoiseVariance
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdModel, noiseModel)


def test_mrw():

    seed(16)

    proposalVariance = 0.01
    mcmc = MetropolisedRandomWalk.from_bayes_model(statModel, proposalVariance)

    # run mcmc
    nSteps = 600
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain

    burnIn = 200
    thinningStep = 3

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


def test_pcn():

    seed(17)

    stepSize = 0.001
    mcmc = PreconditionedCrankNicolson.from_bayes_model(statModel, stepSize)

    # run mcmc
    nSteps = 600
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain

    burnIn = 200
    thinningStep = 3

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


if __name__ == "__main__":
    pytest.main()
