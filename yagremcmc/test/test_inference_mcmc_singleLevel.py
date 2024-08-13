import pytest
import yagremcmc.test.testSetup as setup

import numpy as np

from numpy.random import seed
from yagremcmc.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from yagremcmc.statistics.parameterLaw import Gaussian
from yagremcmc.statistics.noise import CentredGaussianIIDNoise
from yagremcmc.statistics.bayesModel import BayesianRegressionModel
from yagremcmc.chain.metropolisedRandomWalk import MRWFactory
from yagremcmc.chain.preconditionedCrankNicolson import PCNFactory
from yagremcmc.model.forwardModel import ForwardModel


def check_mean(means, trueParam):

    MTOL = 1e-1

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
parameterDim = 2
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))
assert parameterDim == groundTruth.dimension

dataNoiseVariance = 0.05
data = setup.generate_synthetic_data(groundTruth, solver, dataNoiseVariance)

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))

priorMargVar = 0.02
priorCovariance = IIDCovarianceMatrix(parameterDim, priorMargVar)

# set up prior
prior = Gaussian(priorMean, priorCovariance)

# define a noise model
noiseVariance = dataNoiseVariance
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdModel, noiseModel)


@pytest.mark.parametrize("mcmcProposal", ["iid", "indep"])
def test_mrw(mcmcProposal):

    seed(16)

    chainFactory = MRWFactory()

    if (mcmcProposal == 'iid'):

        proposalMargVar = 0.02
        proposalCov = IIDCovarianceMatrix(parameterDim, proposalMargVar)

    elif (mcmcProposal == 'indep'):

        proposalMargVar = np.array([0.02, 0.01])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise Exception("Proposal " + mcmcProposal + " not implemented")

    chainFactory.set_proposal_covariance(proposalCov)
    chainFactory.set_bayes_model(statModel)

    mcmc = chainFactory.build_method()

    # run mcmc
    nSteps = 1000
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain.trajectory

    burnIn = 200
    thinningStep = 5

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


def test_pcn():

    seed(17)

    chainFactory = PCNFactory()
    chainFactory.set_step_size(0.001)
    chainFactory.set_bayes_model(statModel)

    mcmc = chainFactory.build_method()

    # run mcmc
    nSteps = 1000
    initState = setup.LotkaVolterraParameter.from_coefficient(
        np.array([-0.6, -0.3]))
    mcmc.run(nSteps, initState)

    states = mcmc.chain.trajectory

    burnIn = 200
    thinningStep = 5

    mcmcSamples = states[burnIn::thinningStep]
    meanState = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(states, axis=0))
    posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
        np.mean(mcmcSamples, axis=0))

    check_mean([meanState, posteriorMean], groundTruth)


if __name__ == "__main__":
    pytest.main()
