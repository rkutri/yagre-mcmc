import numpy as np
import matplotlib.pyplot as plt

from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.am import AMBuilder
from yagremcmc.parameter.vector import ParameterVector


# current options are 'mrw', 'am'
method = 'am'

# current options are 'iid', 'indep'
mcmcProposal = 'iid'

tgtMean = ParameterVector(np.array([1., 1.5]))
tgtCov = np.array(
    [[2.2, -0.5],
     [-0.5, 0.3]])
tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

if (mcmcProposal == 'iid'):

    proposalMargVar = 0.5
    proposalCov = IIDCovarianceMatrix(tgtMean.dimension, proposalMargVar)

elif (mcmcProposal == 'indep'):

    proposalMargVar = np.array([tgtCov[0, 0], tgtCov[1, 1]])
    proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

else:
    raise Exception("Proposal " + mcmcProposal + " not implemented")

assert method in ('mrw', 'am')
if method == 'mrw':

    chainBuilder = MRWBuilder()

    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.proposalCovariance = proposalCov

else:

    chainBuilder = AMBuilder()

    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.idleSteps = 100
    chainBuilder.collectionSteps = 200
    chainBuilder.regularisationParameter = 1e-4
    chainBuilder.initialCovariance = proposalCov

mcmc = chainBuilder.build_method()

nSteps = 50000
initState = ParameterVector(np.array([-8., -7.]))
mcmc.run(nSteps, initState)

states = np.array(mcmc.chain.trajectory)

# postprocessing
burnin = int(0.001 * nSteps)
thinningStep = 8

mcmcSamples = states[burnin::thinningStep]

# estimate mean
meanState = np.mean(states, axis=0)
meanEst = np.mean(mcmcSamples, axis=0)

print("true mean: " + str(tgtMean.coefficient))
print("mean state: " + str(meanState))
print("mean estimate: " + str(meanEst))
print("acceptance rate: " + str(mcmc.diagnostics.global_acceptance_rate()))

# plotting
xGrid = np.linspace(-8., 8., 200)
yGrid = np.linspace(-8., 8., 200)

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

if method == 'am':
    adaptStart = chainBuilder.idleSteps + chainBuilder.collectionSteps - 1
    plt.scatter(chainX[adaptStart], chainY[adaptStart], color='green',
                marker='x', s=100, label='start of adaptive covariance')

# Enhance the plot
plt.title('2D Markov Chain Path with Target Distribution Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, which='both', linestyle='--',
         linewidth=0.5, color='gray', alpha=0.7)

# Show the plot
plt.show()
