import numpy as np
import matplotlib.pyplot as plt
import yagremcmc.postprocessing.autocorrelation as ac

from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix, DiagonalCovarianceMatrix
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.deprecated.awm import AWMBuilder
from yagremcmc.chain.diagnostics import FullDiagnostics
from yagremcmc.parameter.vector import ParameterVector


# current options are 'mrw', 'awm'
method = 'mrw'

# current options are 'iid', 'indep'
mcmcProposal = 'iid'

tgtMean = ParameterVector(np.array([1., 1.5]))
tgtCov = np.array(
    [[2.4, -0.5],
     [-0.5, 0.7]])
tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

if (mcmcProposal == 'iid'):

    proposalMargVar = 1.0
    proposalCov = IIDCovarianceMatrix(tgtMean.dimension, proposalMargVar)

elif (mcmcProposal == 'indep'):

    proposalMargVar = np.array([tgtCov[0, 0], tgtCov[1, 1]])
    proposalCov = DiagonalCovarianceMatrix(proposalMargVar)

else:
    raise Exception("Proposal " + mcmcProposal + " not implemented")

assert method in ('mrw', 'awm')
if method == 'mrw':

    chainBuilder = MRWBuilder()

    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.proposalCovariance = proposalCov
    chainBuilder.diagnostics = FullDiagnostics()

else:

    chainBuilder = AWMBuilder()

    chainBuilder.explicitTarget = tgtDensity
    chainBuilder.idleSteps = 5000
    chainBuilder.collectionSteps = 5000
    chainBuilder.initialCovariance = proposalCov

mcmc = chainBuilder.build_method()

nSteps = 100000
initState = ParameterVector(np.array([-8., -7.]))
mcmc.run(nSteps, initState)

states = np.array(mcmc.chain.trajectory)

# postprocessing
dim = tgtMean.dimension
burnin = 1000

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

plt.title('2D Markov Chain Path with Target Distribution Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, which='both', linestyle='--',
         linewidth=0.5, color='gray', alpha=0.7)

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
