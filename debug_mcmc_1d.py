import numpy as np
import matplotlib.pyplot as plt

from testSetup import GaussianTargetDensity1d
from inference.metropolisedRandomWalk import MetropolisedRandomWalk
from parameter.scalar import ScalarParameter


tgtMean = ScalarParameter(np.array([1.5]))
tgtVar = 1.
tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

mesh = np.linspace(-8., 8., 1000, endpoint=True)

# evaluate target density and normalise
tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
tgtDensityEval /= np.sqrt(2. * np.pi * tgtVar)


proposalVariance = 0.5
mcmc = MetropolisedRandomWalk(tgtDensity, proposalVariance)

nSteps = 3000
initState = ScalarParameter(np.array([-7.]))
mcmc.run(nSteps, initState, verbose=False)

states = np.array(mcmc.chain)

# postprocessing
burnin = int(0.02 * nSteps)
thinningStep = 4

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanState = np.mean(mcmcSamples)
rawMean = np.mean(states)
print("true mean: " + str(tgtMean.vector[0]))
print("best estimate: " + str(meanState))
print("unprocessed estimate: " + str(rawMean))

plt.hist(states, bins=50, edgecolor='white', alpha=0.4,
         color='red', density=True, label='mc states')
plt.hist(mcmcSamples, bins=50, edgecolor='black', alpha=0.8,
         color='blue', density=True, label='thinned states')
plt.plot(mesh, tgtDensityEval, color='red', label='target density')
plt.legend()

plt.show()
