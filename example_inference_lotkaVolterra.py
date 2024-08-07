import yagremcmc.test.testSetup as setup

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from yagremcmc.model.forwardModel import ForwardModel
from yagremcmc.inference.parameterLaw import IIDGaussian
from yagremcmc.inference.noise import CentredGaussianIIDNoise
from yagremcmc.inference.bayesModel import BayesianRegressionModel
from yagremcmc.inference.metropolisedRandomWalk import MetropolisedRandomWalk
from yagremcmc.inference.preconditionedCrankNicolson import PreconditionedCrankNicolson

# current options are 'mrw', 'pcn'
mcmcMethod = 'pcn'

# define model problem
config = {'T': 10., 'alpha': 0.8, 'gamma': 0.4, 'nData': 10, 'dataDim': 2}
design = np.array([uniform(0.5, 1.5, 2) for _ in range(config['nData'])])

# define forward problem
solver = setup.LotkaVolterraSolver(design, config)
fwdModel = ForwardModel(solver)

# define problem parameters
groundTruth = setup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))

# generate data
dataNoiseVar = 0.04
data = setup.generate_synthetic_data(groundTruth, solver, dataNoiseVar)

print("synthetic data generated")

# start with a prior centred around the true parameter coefficient
priorMean = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))
priorVariance = 1.
prior = IIDGaussian(priorMean, priorVariance)

# define a noise model
noiseVariance = dataNoiseVar
noiseModel = CentredGaussianIIDNoise(noiseVariance)

# define the statistical inverse problem
statModel = BayesianRegressionModel(data, prior, fwdModel, noiseModel)

if (mcmcMethod == 'mrw'):

    proposalVariance = 0.02
    mcmc = MetropolisedRandomWalk.from_bayes_model(statModel, proposalVariance)

elif (mcmcMethod == 'pcn'):

    if (not np.allclose(priorMean.coefficient, np.zeros(2))):
        raise Exception("PCN requires centred prior.")

    stepSize = 0.02
    mcmc = PreconditionedCrankNicolson.from_bayes_model(statModel, stepSize)

else:
    raise Exception("Unknown MCMC method: " + str(mcmcMethod))

# run mcmc
nSteps = 1000
initState = setup.LotkaVolterraParameter.from_coefficient(np.zeros(2))
mcmc.run(nSteps, initState)

states = mcmc.chain

burnIn = 100
thinningStep = 3

mcmcSamples = states[burnIn::thinningStep]
meanState = setup.LotkaVolterraParameter.from_coefficient(
    np.mean(states, axis=0))
posteriorMean = setup.LotkaVolterraParameter.from_coefficient(
    np.mean(mcmcSamples, axis=0))

# estimates mean
print("true parameter: " + str(groundTruth.evaluate()))
print("raw posterior mean: " + str(meanState.evaluate()))
print("processed posterior mean: " + str(posteriorMean.evaluate()))

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
