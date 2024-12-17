import numpy as np
import matplotlib.pyplot as plt
import yagremcmc.postprocessing.autocorrelation as ac

from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.parameter.vector import ParameterVector

tgtMeanCoeff = np.array([1., 1.5])

tgtMean = ParameterVector(tgtMeanCoeff)
# baseSurrMean = tgtMean
# fineSurrMean = tgtMean

baseSurrMean = ParameterVector(tgtMeanCoeff + np.array([-0.05, 0.01]))
fineSurrMean = ParameterVector(tgtMeanCoeff + np.array([0., -0.01]))

tgtCov = np.array(
    [[2.4, -0.5],
     [-0.5, 0.7]])

# baseSurrCov = tgtCov
# fineSurrCov = tgtCov

baseSurrCov = 4. * np.array(
    [[2.8, -0.1],
     [-0.1, 1.7]])
fineSurrCov = 2 * np.array(
    [[2.4, -0.3],
     [-0.3, 1.1]])

tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)
baseSurrDensity = GaussianTargetDensity2d(baseSurrMean, baseSurrCov)
fineSurrDensity = GaussianTargetDensity2d(fineSurrMean, fineSurrCov)

basePropMargVar = 1.
basePropCov = IIDCovarianceMatrix(tgtMean.dimension, basePropMargVar)

chainBuilder = MLDABuilder()

chainBuilder.explicitTarget = tgtDensity
chainBuilder.surrogateTargets = [baseSurrDensity, fineSurrDensity]
chainBuilder.baseProposalCovariance = basePropCov
chainBuilder.subChainLengths = [4, 4]

mcmc = chainBuilder.build_method()

nSteps = 50000
initState = ParameterVector(np.array([-8., -7.]))
mcmc.run(nSteps, initState)

states = np.array(mcmc.chain.trajectory)

# postprocessing
dim = tgtMean.dimension
burnin = 100

assert nSteps > burnin

# estimate autocorrelation function
acf = [ac.estimate_autocorrelation_function_1d(
    states[burnin:, d]) for d in range(dim)]

meanIAT = ac.integrated_autocorrelation(states[burnin:], 'mean')
maxIAT = ac.integrated_autocorrelation(states[burnin:], 'max')

thinningStep = maxIAT

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanState = np.mean(states, axis=0)
meanEst = np.mean(mcmcSamples, axis=0)

print("\nAnalytics")
print("---------")
print(f"acceptance rate: {mcmc.diagnostics.global_acceptance_rate()}")
print(f"mean IAT: {meanIAT}")
print(f"max IAT: {maxIAT}\n")

print("Inference")
print("---------")
print(f"true mean: {tgtMean.coefficient}")
print(f"mean state: {meanState}")
print(f"mean estimate: {meanEst}")

# plotting
xGrid = np.linspace(-8., 8., 400)
yGrid = np.linspace(-8., 8., 400)

# Create a grid for the contour plot
X, Y = np.meshgrid(xGrid, yGrid)
mesh = np.dstack((X, Y))

tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
baseDensityEval = baseSurrDensity.evaluate_on_mesh(mesh)
fineDensityEval = fineSurrDensity.evaluate_on_mesh(mesh)

# Plotting
plt.figure(figsize=(8, 6))

# Plot the contour lines of the target distribution
plt.contour(X, Y, tgtDensityEval, levels=4, cmap='Reds')
plt.contour(X, Y, baseDensityEval, levels=4, cmap='Blues')
plt.contour(X, Y, fineDensityEval, levels=4, cmap='Greens')

# Extract x and y coordinates
chainX = [state[0] for state in states]
chainY = [state[1] for state in states]

mcmcX = [sample[0] for sample in mcmcSamples]
mcmcY = [sample[1] for sample in mcmcSamples]


# Plot the Markov chain trajectory
plt.title('2D Markov Chain Path with Target Distribution Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, which='both', linestyle='--',
         linewidth=0.5, color='gray', alpha=0.7)

plt.scatter(mcmcX, mcmcY, color='gray', marker='o', s=40,
            alpha=0.1, label='selected samples')
plt.scatter(meanEst[0], meanEst[1], color='black', s=100,
            marker='P', label='mcmc mean estimate')

plt.show()

# plot autocorrelation functions
plt.title('Autocorrelation Functions')
plt.xlabel('lag')
plt.ylabel('estimated correlation')

plt.semilogx(
    np.arange(1, nSteps + 1 - burnin), acf[0],
    label="estimated ACF, coordinate 0")
plt.semilogx(
    np.arange(1, nSteps + 1 - burnin), acf[1],
    label="estimated ACF, coordinate 1")
plt.xlim(1, nSteps + 1 - burnin)
plt.ylim(-0.15, 1.)

plt.axvline(meanIAT, color='r', linestyle='--', label=f"mean IAT")
plt.axvline(maxIAT, color='b', linestyle='--', label=f"max IAT")
plt.legend()

plt.show()
