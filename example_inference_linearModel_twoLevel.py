import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed, standard_normal
from exampleSetup import ExampleLinearModelSolver, evaluate_posterior
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.model.forwardModel import ForwardModel
from yagremcmc.statistics.data import Data
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.statistics.gaussian import Gaussian
from yagremcmc.statistics.noise import CentredGaussianNoise
from yagremcmc.statistics.likelihood import (AdditiveGaussianNoiseLikelihood,
                                             AEMLikelihood)
from yagremcmc.statistics.bayesModel import BayesianRegressionModel
from yagremcmc.statistics.modelHierarchy import BayesianRegressionModelHierarchy
from yagremcmc.utility.hierarchy import SharedComponent, Hierarchy
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.chain.method.aem import AEMBuilder
from yagremcmc.postprocessing.autocorrelation import integrated_autocorrelation


seed(2222)
DIM = 2


# -----------------------------------------------------------------------------
#                             PROBLEM SETUP
# -----------------------------------------------------------------------------

# set up explicit linear models
tgtMap = np.array([[1.4, -0.2], [-0.6, 0.7]])
tgtShift = np.zeros(DIM)
tgtSolver = ExampleLinearModelSolver(tgtMap, tgtShift)

surMapError = np.array([[-0.6, -0.2], [0.4, 1.1]])
surMap = tgtMap + surMapError
surShift = np.array([0.5, -0.9])
surSolver = ExampleLinearModelSolver(surMap, surShift)

tgtFwdModel = ForwardModel(tgtSolver)
surFwdModel = ForwardModel(surSolver)

# generate data
trueParam = ParameterVector(np.array([1.5, 0.5]))
dataNoiseStdDev = np.sqrt(0.3)
nData = 5

tgtSolver.interpolate(trueParam)
tgtSolver.invoke()

data = Data(np.array(
    [tgtSolver.evaluation + dataNoiseStdDev * standard_normal(DIM)
        for _ in range(nData)]))

assert data.size == nData
assert data.dim == DIM


# -----------------------------------------------------------------------------
#                       BAYES MODEL DEFINITIONS
# -----------------------------------------------------------------------------

# PRIOR
# -----

priorMeanError = np.array([-0.2, 0.4])
priorMean = ParameterVector(trueParam.coefficient + priorMeanError)

priorMargVar = 5.
priorCovariance = IIDCovarianceMatrix(DIM, priorMargVar)

prior = Gaussian(priorMean, priorCovariance)


# NOISE
# -----

noiseMargVar = dataNoiseStdDev**2
noiseCov = IIDCovarianceMatrix(DIM, noiseMargVar)
noiseModel = CentredGaussianNoise(noiseCov)


# LIKELIHOOD
# ----------

vanillaSurLikelihood = AdditiveGaussianNoiseLikelihood(
    data, surFwdModel, noiseModel)
vanillaTgtLikelihood = AdditiveGaussianNoiseLikelihood(
    data, tgtFwdModel, noiseModel)

vanillaLikelihood = [vanillaSurLikelihood, vanillaTgtLikelihood]

minDataSize = 100
aemSurLikelihood = AEMLikelihood(
    data, surFwdModel, noiseModel, minDataSize)
aemTgtLikelihood = AEMLikelihood(
    data, tgtFwdModel, noiseModel, minDataSize)

aemLikelihood = [aemSurLikelihood, aemTgtLikelihood]


# MODEL
# -----

nLevels = 2

# declare shared and hierarchical components of the model
prior = SharedComponent(prior, nLevels)
noiseModel = SharedComponent(noiseModel, nLevels)
vanillaLikelihood = Hierarchy(vanillaLikelihood)
aemLikelihood = Hierarchy(aemLikelihood)

# build models
tgtModel = BayesianRegressionModel(vanillaTgtLikelihood, prior.level(0))
surModel = BayesianRegressionModel(vanillaSurLikelihood, prior.level(0))

vanillaModel = BayesianRegressionModelHierarchy(vanillaLikelihood, prior)
aemModel = BayesianRegressionModelHierarchy(aemLikelihood, prior)


# -----------------------------------------------------------------------------
#                             MCMC SETUP
# -----------------------------------------------------------------------------

proposalMVar = 0.5
proposalCov = IIDCovarianceMatrix(DIM, proposalMVar)

# Target MRW
# -------------

tgtBuilder = MRWBuilder()

tgtBuilder.proposalCovariance = proposalCov
tgtBuilder.bayesModel = tgtModel

print("\nrequest building target mrw")
tgtMRW = tgtBuilder.build_method()

# Surrogate MRW
# -------------

surBuilder = MRWBuilder()

surBuilder.proposalCovariance = proposalCov
surBuilder.bayesModel = surModel

print("\nrequest building surrogate mrw")
surMRW = surBuilder.build_method()


# MLDA Burn-In Chain
# ------------------

mldaBuilder = MLDABuilder()

mldaBuilder.baseProposalCovariance = proposalCov
mldaBuilder.subChainLengths = [2]
mldaBuilder.bayesModel = vanillaModel

print("\nrequest building burn-in mlda")
mldaBurnin = mldaBuilder.build_method()


# vanilla MLDA
# ------------

mldaBuilder = MLDABuilder()

mldaBuilder.baseProposalCovariance = proposalCov
mldaBuilder.subChainLengths = [5]
mldaBuilder.bayesModel = vanillaModel

print("\nrequest building vanilla mlda")
vanillaMLDA = mldaBuilder.build_method()


# adaptive error model MLDA
# ------------------------------

aemBuilder = AEMBuilder()

aemBuilder.baseProposalCovariance = proposalCov
aemBuilder.subChainLengths = [5]
aemBuilder.bayesModel = aemModel

print("\nrequest building aem mlda")
aemMLDA = aemBuilder.build_method()


# -----------------------------------------------------------------------------
#                                INFERENCE
# -----------------------------------------------------------------------------

tgtNSteps = 50000
tgtBurnin = 500

initState = ParameterVector(np.zeros(DIM))

print(f"\nRunning {tgtNSteps} steps of the target MRW")

tgtMRW.run(tgtNSteps, initState)
tgtStates = tgtMRW.chain.trajectory

tgtThinning = integrated_autocorrelation(tgtStates, 'mean')
tgtMean = np.mean(tgtStates[tgtBurnin::tgtThinning], axis=0)

tgtAccPr = tgtMRW.diagnostics.global_acceptance_rate()

surNSteps = 50000
surBurnin = 500

initState = ParameterVector(np.zeros(DIM))

print(f"\nRunning {surNSteps} steps of the surrogate MRW")

surMRW.run(surNSteps, initState)
surStates = surMRW.chain.trajectory

surThinning = integrated_autocorrelation(surStates, 'mean')
surMean = np.mean(surStates[surBurnin::surThinning], axis=0)
surAccPr = surMRW.diagnostics.global_acceptance_rate()

# start the actual burn-in chain where the surrogate chain left off
initState = ParameterVector(surStates[-1])
nSteps = 500

print(f"\n\n\nBurning-in MLDA with {nSteps} steps, starting at last surrogate chain "
      "position")

mldaBurnin.run(nSteps, initState)

initState = ParameterVector(mldaBurnin.chain.trajectory[-1])
initStateCopy = np.copy(initState)

nSteps = 50000

print(f"\n\n\nRunning {nSteps} steps of vanilla MLDA")

vanillaMLDA.run(nSteps, initState)

vMLDAStates = vanillaMLDA.chain.trajectory
vMLDAThinning = integrated_autocorrelation(vMLDAStates, 'max')
vMLDAPostMean = np.mean(vMLDAStates[::vMLDAThinning], axis=0)
vMLDAAccPr = vanillaMLDA.diagnostics.global_acceptance_rate()

print(f"\n\n\nRunning {nSteps} steps of MLDA with an adaptive error model")

nSteps = 50000
assert initState == initStateCopy

aemMLDA.run(nSteps, initState)

aMLDAStates = aemMLDA.chain.trajectory
aMLDAThinning = integrated_autocorrelation(aMLDAStates, 'max')
aMLDAPostMean = np.mean(aMLDAStates[::aMLDAThinning], axis=0)
aMLDAAccPr = aemMLDA.diagnostics.global_acceptance_rate()

print("\n\nResults for target MRW")
print("----------------------")
print(f"\nacceptance rate: {tgtAccPr}")
print(f"IAT estimate: {tgtThinning}")
print(f"posterior mean: {tgtMean}")

print(f"Likelihood cache hits: {vanillaSurLikelihood._llCache.hits}")
print(f"Likelihood cache misses: {vanillaSurLikelihood._llCache.misses}")

print("\n\nResults for surrogate MRW")
print("----------------------")
print(f"\nacceptance rate: {surAccPr}")
print(f"IAT estimate: {surThinning}")
print(f"posterior mean: {surMean}")

print("\n\nResults for vanilla MLDA")
print("-------------------------")
print(f"acceptance rate: {vMLDAAccPr}")
print(f"IAT estimate: {vMLDAThinning}")
print(f"posterior mean: {vMLDAPostMean}")

print(f"\n\nResults for AEM MLDA")
print("---------------------")
print(f"acceptance rate: {aMLDAAccPr}")
print(f"IAT estimate: {aMLDAThinning}")
print(f"posterior mean: {aMLDAPostMean}")
print(f"estimated mean error: {aemSurLikelihood.accumulator.mean()}")

print("\nAEM DETAILS:")

print("Total number of carried out surrogate evaluations: "
      f"{aemSurLikelihood.number_of_model_evaluations()}")
print("Total number of carried out target evaluations: "
      f"{aemTgtLikelihood.number_of_model_evaluations()}")

print(f"surrogate cache hits: {aemSurLikelihood._cache.hits}")
print(f"surrogate cache misses: {aemSurLikelihood._cache.misses}")

print(f"target cache hits: {aemTgtLikelihood._cache.hits}")
print(f"target cache misses: {aemTgtLikelihood._cache.misses}")

if aemSurLikelihood.accumulator.nData > 1:
    print("estimated error marginal variance: "
          f"{aemSurLikelihood.accumulator.marginal_variance()}")

# -----------------------------------------------------------------------------
#                                PLOTTING
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(1, 4, figsize=(12, 6))

xGrid = np.linspace(-3., 4., 100)
yGrid = np.linspace(-1., 2., 100)

# Create a grid for the contour plot
X, Y = np.meshgrid(xGrid, yGrid)
mesh = np.dstack((X, Y))


# --------------------- PLOT 1: TARGET MRW CHAIN ------------------------------

tgtDensityEval = evaluate_posterior(mesh, vanillaTgtLikelihood, prior.level(0))
ax[0].contour(X, Y, tgtDensityEval, levels=4, cmap='Blues')

tgtMRWSamples = tgtMRW.chain.trajectory[tgtBurnin::tgtThinning]

burninX = [state[0] for state in tgtMRW.chain.trajectory[:tgtBurnin]]
burninY = [state[1] for state in tgtMRW.chain.trajectory[:tgtBurnin]]

mcmcX = [sample[0] for sample in tgtMRWSamples]
mcmcY = [sample[1] for sample in tgtMRWSamples]

# Plot the Markov chain trajectory
# ax[0].plot(burninX, burninY, color='green', alpha=0.1, label='burn-in')
ax[0].scatter(burninX, burninY, color='green', alpha=0.2)
ax[0].scatter(
    mcmcX, mcmcY,
    color='gray', marker='o', s=40, alpha=0.2,
    label='selected samples')
ax[0].scatter(tgtMean[0], tgtMean[1], color='black', s=100,
              marker='P', label='estimated tgtrogate mean')
# Add labels, legend, and grid
ax[0].set_title('Target MRW Chain', fontsize=24)
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend(fontsize=14)
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5,
           color='gray', alpha=0.7)

# Add acceptance rate and IAT
ax[0].text(0.55, 0.9,
           f"acceptance prob.: {tgtAccPr:.3f}\nIAT estimate: {tgtThinning}",
           transform=ax[0].transAxes, ha='right', va='top', fontsize=20,
           color='black', bbox=dict(facecolor='white',
                                    edgecolor='k',
                                    alpha=0.7))


# ----------------- PLOT 2: SURROGATE CHAIN -----------------------------------

surDensityEval = evaluate_posterior(mesh, vanillaSurLikelihood, prior.level(0))
ax[1].contour(X, Y, surDensityEval, levels=4, cmap='Reds')

surMRWSamples = surMRW.chain.trajectory[surBurnin::surThinning]

burninX = [state[0] for state in surMRW.chain.trajectory[:surBurnin]]
burninY = [state[1] for state in surMRW.chain.trajectory[:surBurnin]]

mcmcX = [sample[0] for sample in surMRWSamples]
mcmcY = [sample[1] for sample in surMRWSamples]

# Plot the Markov chain trajectory
# ax[1].plot(burninX, burninY, color='green', alpha=0.1, label='burn-in')
ax[1].scatter(burninX, burninY, color='green', alpha=0.2)
ax[1].scatter(
    mcmcX, mcmcY,
    color='gray', marker='o', s=40, alpha=0.2,
    label='selected samples')
ax[1].scatter(surMean[0], surMean[1], color='black', s=100,
              marker='P', label='estimated surrogate mean')

# Add labels, legend, and grid
ax[1].set_title('Surrogate Chain', fontsize=24)
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
ax[1].legend(fontsize=14)
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5,
           color='gray', alpha=0.7)

# Add acceptance rate and IAT
ax[1].text(0.55, 0.9, f"acceptance prob.: {surAccPr:.3f}\nIAT estimate: {surThinning}",
           transform=ax[1].transAxes, ha='right', va='top', fontsize=20, color='black',
           bbox=dict(facecolor='white', edgecolor='k', alpha=0.7))


# -------------------- PLOT 3: VANILLA MLDA -----------------------------------

tgtDensityEval = evaluate_posterior(mesh, vanillaTgtLikelihood, prior.level(0))

ax[2].contour(X, Y, surDensityEval, levels=4, cmap='Reds')
ax[2].contour(X, Y, tgtDensityEval, levels=4, cmap='Blues')

vMLDASamples = vMLDAStates[::vMLDAThinning]

# Extract x and y coordinates
burninX = [state[0] for state in mldaBurnin.chain.trajectory]
burninY = [state[1] for state in mldaBurnin.chain.trajectory]
mcmcX = [sample[0] for sample in vMLDASamples]
mcmcY = [sample[1] for sample in vMLDASamples]

# Plot the Markov chain trajectory
# ax[2].scatter(burninX, burninY, color='green', alpha=0.2)
ax[2].scatter(
    mcmcX, mcmcY,
    color='gray', marker='o', s=40, alpha=0.2,
    label='selected samples')
ax[2].scatter(vMLDAPostMean[0], vMLDAPostMean[1], color='black', s=100,
              marker='P', label='estimated posterior mean')

# Add labels, legend, and grid
ax[2].set_title('Vanilla MLDA', fontsize=24)
ax[2].set_xlabel('X')
ax[2].set_ylabel('Y')
ax[2].legend(fontsize=14)
ax[2].grid(True, which='both', linestyle='--', linewidth=0.5,
           color='gray', alpha=0.7)

# Add acceptance rate and IAT
ax[2].text(0.55, 0.9, f"acceptance prob.: {vMLDAAccPr:.3f}\nIAT estimate: {vMLDAThinning}",
           transform=ax[2].transAxes, ha='right', va='top', fontsize=20, color='black',
           bbox=dict(facecolor='white', edgecolor='k', alpha=0.7))

# ------------------------ PLOT 4: AEM MLDA -----------------------------------

corrTgtDensityEval = evaluate_posterior(mesh, aemTgtLikelihood, prior.level(0))
corrSurDensityEval = evaluate_posterior(mesh, aemSurLikelihood, prior.level(0))

ax[3].contour(X, Y, corrTgtDensityEval, levels=4, cmap='Blues')
ax[3].contour(X, Y, corrSurDensityEval, levels=4, cmap='Reds')

aMLDASamples = aMLDAStates[::aMLDAThinning]

# Extract x and y coordinates
burninX = [state[0] for state in mldaBurnin.chain.trajectory]
burninY = [state[1] for state in mldaBurnin.chain.trajectory]
mcmcX = [sample[0] for sample in aMLDASamples]
mcmcY = [sample[1] for sample in aMLDASamples]

# Plot the Markov chain trajectory
# ax[3].scatter(burninX, burninY, color='green', alpha=0.2)
ax[3].scatter(
    mcmcX, mcmcY,
    color='gray', marker='o', s=40, alpha=0.2,
    label='selected samples')
ax[3].scatter(aMLDAPostMean[0], aMLDAPostMean[1], color='black', s=100,
              marker='P', label='estimated posterior mean')

# Add labels, legend, and grid
ax[3].set_title('MLDA with AEM', fontsize=24)
ax[3].set_xlabel('X')
ax[3].set_ylabel('Y')
ax[3].legend(fontsize=14)
ax[3].grid(True, which='both', linestyle='--', linewidth=0.5,
           color='gray', alpha=0.7)

# Add acceptance rate and IAT
ax[3].text(0.55, 0.9, f"acceptance prob.: {aMLDAAccPr:.3f}\nIAT estimate: {aMLDAThinning}",
           transform=ax[3].transAxes, ha='right', va='top', fontsize=20, color='black',
           bbox=dict(facecolor='white', edgecolor='k', alpha=0.7))

plt.show()
