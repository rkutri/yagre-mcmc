import yagremcmc.test.testSetup as setup

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from yagremcmc.model.forwardModel import ForwardModel
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.pcn import PCNBuilder
from yagremcmc.chain.method.am import AMBuilder
from yagremcmc.statistics.parameterLaw import Gaussian
from yagremcmc.statistics.covariance import DiagonalCovarianceMatrix, IIDCovarianceMatrix
from yagremcmc.statistics.noise import CentredGaussianIIDNoise
from yagremcmc.statistics.bayesModel import BayesianRegressionModel

# available options are 'mrw', 'pcn', 'am'
method = 'am'

if method != 'pcn':

    # available options are 'iid', 'indep'. For 'am',
    # this will be used as the initial covariance.
    proposalCovType = 'iid'

# define model problem
config = {'T': 10., 'alpha': 0.8, 'gamma': 0.4, 'nData': 10, 'dataDim': 2}
design = np.array([uniform(0.5, 1.5, 2) for _ in range(config['nData'])])

# define forward problem
solver = setup.LotkaVolterraSolver(design, config)
fwdModel = ForwardModel(solver)

# define problem parameters
parameterDim = 2
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))
assert groundTruth.dimension == parameterDim

# generate data
dataNoiseVar = 0.04
data = setup.generate_synthetic_data(groundTruth, solver, dataNoiseVar)

print("synthetic data generated")

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))

priorMargVar = 1.2
priorCovariance = IIDCovarianceMatrix(parameterDim, priorMargVar)

# set up prior
prior = Gaussian(priorMean, priorCovariance)

# define a noise model
noiseVariance = dataNoiseVar
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdModel, noiseModel)

# configure the chain setup
if method == 'pcn':

    chainBuilder = PCNBuilder()
    chainBuilder.stepSize = 0.01

elif method in ['mrw', 'am']:

    if proposalCovType == 'iid':

        proposalMargVar = 0.2
        proposalCov = IIDCovarianceMatrix(parameterDim, proposalMargVar)

    elif proposalCovType == 'indep':

        proposalMargVar = np.array([0.02, 0.06])
        proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

    else:
        raise ValueError(
            "Unknown Proposal covariance type: " + proposalCovType)

    if method == 'mrw':

        chainBuilder = MRWBuilder()
        chainBuilder.proposalCovariance = proposalCov

    elif method == 'am':

        chainBuilder = AMBuilder()

        chainBuilder.idleSteps = 50
        chainBuilder.collectionSteps = 100
        chainBuilder.regularisationParameter = 1e-6
        chainBuilder.initialCovariance = proposalCov

    else:
        raise ValueError("Unknown MCMC method: " + method)


else:
    raise ValueError("Unknown MCMC method: " + method)

chainBuilder.bayesModel = statModel

sampler = chainBuilder.build_method()

# run mcmc
nSteps = 5000
initState = setup.LotkaVolterraParameter.from_coefficient(np.array([-7., 2.8]))
sampler.run(nSteps, initState)

states = sampler.chain.trajectory

burnIn = 100
thinningStep = 6

mcmcSamples = states[burnIn::thinningStep]
meanState = setup.LotkaVolterraParameter.from_coefficient(
    np.mean(states, axis=0))
posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
    np.mean(mcmcSamples, axis=0))

# estimates mean
print("true parameter: " + str(groundTruth.evaluate()))
print("raw posterior mean: " + str(meanState.evaluate()))
print("processed posterior mean: " + str(posteriorMean.evaluate()))
print("Acceptance rate: " + str(sampler.chain.diagnostics.global_acceptance_rate()))

# Plotting
fig, ax = plt.subplots(1, 2)
plt.rcParams["figure.figsize"] = (8, 6)

# Extract x and y coordinates
chainX = [state[0] for state in states]
chainY = [state[1] for state in states]

mcmcX = [sample[0] for sample in mcmcSamples]
mcmcY = [sample[1] for sample in mcmcSamples]

# Plot the Markov chain trajectory
ax[0].plot(chainX[:burnIn], chainY[:burnIn], color='gray', alpha=0.4,
           label='burn-in')
ax[0].scatter(chainX, chainY, color='red', marker='o', alpha=0.1, s=80,
              label='mc states')
ax[0].scatter(mcmcX, mcmcY, color='blue', marker='o', s=80,
              alpha=0.6, label='selected samples')
ax[0].scatter(
    posteriorMean.coefficient[0],
    posteriorMean.coefficient[1],
    color='green',
    marker='P',
    label='posterior mean',
    s=120)
ax[0].scatter(
    meanState.coefficient[0],
    meanState.coefficient[1],
    color='black',
    marker='P',
    label='markov chain mean',
    s=120)
ax[0].scatter(
    groundTruth.coefficient[0],
    groundTruth.coefficient[1],
    color='red',
    marker='P',
    label='true parameter',
    s=120)

if method == 'am':
    adaptStart = chainBuilder.idleSteps + chainBuilder.collectionSteps - 1
    ax[0].scatter(chainX[adaptStart], chainY[adaptStart], color='green',
                  marker='x', s=100, label='start of adaptive covariance')

ax[0].set_title('2D Markov Chain Path')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--',
           linewidth=0.5, color='gray', alpha=0.7)

trueSol = solver.full_solution(groundTruth, np.array([1., 1.]))
tGridSol = trueSol[0]
xSol = trueSol[1][0, :]
ySol = trueSol[1][1, :]

estSol = solver.full_solution(posteriorMean, np.array([1., 1.]))
tGridEst = estSol[0]
xEst = estSol[1][0, :]
yEst = estSol[1][1, :]

ax[1].plot(tGridSol, xSol, label='true sol x', color='red')
ax[1].plot(tGridSol, ySol, label='true sol y', color='orange')
ax[1].plot(tGridEst, xEst, label='estimated x', color='red', linestyle='--')
ax[1].plot(tGridEst, yEst, label='estimated y', color='orange', linestyle='--')
ax[1].legend()

plt.show()
