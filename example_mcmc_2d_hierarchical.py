import numpy as np
import matplotlib.pyplot as plt

from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.parameter.vector import ParameterVector


tgtMean = ParameterVector(np.array([1., 1.5]))
surrMean = tgtMean

tgtCov = np.array(
    [[2.4, -0.5],
     [-0.5, 0.7]])
surrCov = np.array(
    [[2.8, 0.1],
     [0.1, 1.7]])

tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)
surrDensity = GaussianTargetDensity2d(surrMean, surrCov)

coarsePropMargVar = 0.25
coarsePropCov = IIDCovarianceMatrix(tgtMean.dimension, coarsePropMargVar)

chainBuilder = MLDABuilder()

chainBuilder.explicitTarget = tgtDensity
chainBuilder.surrogateTargets = [surrDensity]
chainBuilder.coarseProposalCovariance = coarsePropCov
chainBuilder.subChainLengths = [5]

mcmc = chainBuilder.build_method()

nSteps = 50000
initState = ParameterVector(np.array([-8., -7.]))
mcmc.run(nSteps, initState)

states = np.array(mcmc.chain.trajectory)

# postprocessing
burnin = 1000
thinningStep = 8

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanState = np.mean(states, axis=0)
meanEst = np.mean(mcmcSamples, axis=0)

print("true mean: " + str(tgtMean.coefficient))
print("mean state: " + str(meanState))
print("mean estimate: " + str(meanEst))
print("acceptance rate: " + str(mcmc.chain.diagnostics.global_acceptance_rate()))

# plotting
xGrid = np.linspace(-8., 8., 400)
yGrid = np.linspace(-8., 8., 400)

# Create a grid for the contour plot
X, Y = np.meshgrid(xGrid, yGrid)
mesh = np.dstack((X, Y))
densityEval = tgtDensity.evaluate_on_mesh(mesh)

# Plotting
plt.figure(figsize=(8, 6))

# Plot the contour lines of the target distribution
plt.contour(X, Y, densityEval, levels=10, cmap='viridis')

# Extract x and y coordinates
chainX = [state[0] for state in states]
chainY = [state[1] for state in states]

mcmcX = [sample[0] for sample in mcmcSamples]
mcmcY = [sample[1] for sample in mcmcSamples]


# Plot the Markov chain trajectory
plt.plot(chainX[:burnin], chainY[:burnin],
         color='gray', alpha=0.6, label='burn-in')
plt.scatter(chainX, chainY, color='red', marker='o', alpha=0.1, s=40,
            label='mc states')
plt.scatter(mcmcX, mcmcY, color='blue', marker='o', s=40,
            alpha=0.5, label='selected samples')
plt.scatter(meanEst[0], meanEst[1], color='black', s=100,
            marker='P', label='mcmc mean estimate')


# Enhance the plot
plt.title('2D Markov Chain Path with Target Distribution Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, which='both', linestyle='--',
         linewidth=0.5, color='gray', alpha=0.7)

# Show the plot
plt.show()
