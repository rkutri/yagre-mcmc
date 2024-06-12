import testSetup

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from inference.parameterLaw import IIDGaussian
from inference.noise import CentredGaussianIIDNoise
from inference.bayes import BayesianRegressionModel
from inference.preconditionedCrankNicolson import PreconditionedCrankNicolson


# define problem parameters
groundTruth = testSetup.LotkaVolterraParameter.from_interpolation(
    np.array([0.4, 0.6]))
alpha = 0.8
gamma = 0.4
T = 15.

# define forward map
fwdMap = testSetup.LotkaVolterraForwardMap(groundTruth, T, alpha, gamma)

# generate the data
dataSize = 5
inputData = [uniform(0.5, 1.5, 2) for _ in range(dataSize)]

dataNoiseVariance = 0.05
data = testSetup.generate_synthetic_data(fwdMap, inputData, dataNoiseVariance)

print("synthetic data generated")

# start with a prior centred around zero. Necessary for pCN
priorMean = testSetup.LotkaVolterraParameter(np.zeros(2))
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
nSteps = 1000
initState = testSetup.LotkaVolterraParameter(np.zeros(2))
mcmc.run(nSteps, initState)

states = mcmc.chain

burnIn = int(0.1 * nSteps)
thinningStep = 6

mcmcSamples = states[burnIn::thinningStep]
meanState = testSetup.LotkaVolterraParameter(np.mean(states, axis=0))
posteriorMean = testSetup.LotkaVolterraParameter(np.mean(mcmcSamples, axis=0))

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
ax[0].scatter(posteriorMean.vector[0], posteriorMean.vector[1], color='green',
              marker='P', label='posterior mean', s=120)
ax[0].scatter(meanState.vector[0], meanState.vector[1], color='black',
              marker='P', label='markov chain mean', s=120)
ax[0].scatter(groundTruth.vector[0], groundTruth.vector[1], color='red',
              marker='P', label='true parameter', s=120)

ax[0].set_title('2D Markov Chain Path')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--',
           linewidth=0.5, color='gray', alpha=0.7)

fwdMap.parameter = groundTruth
trueSol = fwdMap.full_solution([1., 1.])
tGridSol = trueSol[0]
xSol = trueSol[1][0, :]
ySol = trueSol[1][1, :]

fwdMap.parameter = posteriorMean
estSol = fwdMap.full_solution([1., 1.])
tGridEst = estSol[0]
xEst = estSol[1][0, :]
yEst = estSol[1][1, :]

ax[1].plot(tGridSol, xSol, label='true sol x', color='red')
ax[1].plot(tGridSol, ySol, label='true sol y', color='orange')
ax[1].plot(tGridEst, xEst, label='estimated x', color='red', linestyle='--')
ax[1].plot(tGridEst, yEst, label='estimated y', color='orange', linestyle='--')
ax[1].legend()

plt.show()
