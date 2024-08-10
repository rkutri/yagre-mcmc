import numpy as np
import matplotlib.pyplot as plt

from yagremcmc.test.testSetup import GaussianTargetDensity1d
from yagremcmc.inference.covariance import IIDCovarianceMatrix
from yagremcmc.inference.metropolisedRandomWalk import MetropolisedRandomWalk
from yagremcmc.parameter.scalar import ScalarParameter


tgtMean = ScalarParameter(np.array([1.5]))
tgtVar = 1.
tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

mesh = np.linspace(-5., 5., 200, endpoint=True)

# evaluate target density and normalise
tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
tgtDensityEval /= np.sqrt(2. * np.pi * tgtVar)

proposalVariance = 0.5
proposalCov = IIDCovarianceMatrix(1, proposalVariance)
mcmc = MetropolisedRandomWalk(tgtDensity, proposalCov)

nSteps = 50000
initState = ScalarParameter(np.array([-3.]))
mcmc.run(nSteps, initState, verbose=False)

states = np.array(mcmc.chain)

# postprocessing
burnin = int(0.02 * nSteps)
thinningStep = 5

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanSample = np.mean(mcmcSamples)
meanState = np.mean(states)

print("true mean: " + str(tgtMean.coefficient[0]))
print("best estimate: " + str(meanSample))
print("unprocessed estimate: " + str(meanState))

plt.hist(states, bins=50, edgecolor='white', alpha=0.4,
         color='red', density=True, label='mc states')
plt.hist(mcmcSamples, bins=50, edgecolor='black', alpha=0.8,
         color='blue', density=True, label='thinned states')
plt.plot(mesh, tgtDensityEval, color='red', label='target density')
plt.legend()

plt.show()
