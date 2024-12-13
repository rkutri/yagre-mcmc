import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed, standard_normal
from exampleSetup import ExampleLinearModelSolver
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.model.forwardModel import ForwardModel
from yagremcmc.statistics.data import Data
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.statistics.gaussian import Gaussian
from yagremcmc.statistics.noise import CentredGaussianIIDNoise
from yagremcmc.statistics.likelihood import (AdditiveGaussianNoiseLikelihood,
                                             AdaptiveErrorCorrectionLikelihood)
from yagremcmc.statistics.bayesModel import BayesianRegressionModel
from yagremcmc.statistics.modelHierarchy import BayesianModelHierarchy
from yagremcmc.utility.hierarchy import SharedComponent, Hierarchy
from yagremcmc.chain.method.mrw import MRWBuilder
from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.chain.method.amlda import AdaptiveErrorCorrectionMLDABuilder
from yagremcmc.postprocessing.autocorrelation import integrated_autocorrelation


seed(2222)
DIM = 2


# -----------------------------------------------------------------------------
#                             PROBLEM SETUP
# -----------------------------------------------------------------------------

# set up explicit linear models
tgtMap = np.array([[2.4, 0.2], [-0.6, 0.4]])
tgtShift = np.zeros(DIM)
tgtSolver = ExampleLinearModelSolver(tgtMap, tgtShift)

surMapError = np.array([[-0.6, -0.1], [0.4, 1.2]])
surMap = tgtMap + surMapError
surShift = np.array([0.3, -0.7])
surSolver = ExampleLinearModelSolver(surMap, surShift)

tgtFwdModel = ForwardModel(tgtSolver)
surFwdModel = ForwardModel(surSolver)

# generate data
trueParam = ParameterVector(np.array([1.5, 0.5]))
dataNoiseMargVar = 0.5
nData = 5

# we're commiting the inverse crime here
tgtSolver.interpolate(trueParam)
tgtSolver.invoke()

data = Data(
    [tgtSolver.evaluation + dataNoiseMargVar * standard_normal(DIM)
        for _ in range(nData)])

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

noiseMargVar = dataNoiseMargVar
noiseModel = CentredGaussianIIDNoise(noiseMargVar)


# LIKELIHOOD
# ----------

vanillaSurLikelihood = AdditiveGaussianNoiseLikelihood(
    data, surFwdModel, noiseModel)
vanillaTgtLikelihood = AdditiveGaussianNoiseLikelihood(
    data, tgtFwdModel, noiseModel)

vanillaLikelihood = [vanillaSurLikelihood, vanillaTgtLikelihood]

minDataSize = 500
aemSurLikelihood = AdaptiveErrorCorrectionLikelihood(
    data, surFwdModel, noiseModel, minDataSize)
aemTgtLikelihood = AdaptiveErrorCorrectionLikelihood(
    data, surFwdModel, noiseModel, minDataSize)

aemLikelihood = [aemSurLikelihood, aemTgtLikelihood]


# MODEL
# -----

nLevels = 2

# declare shared and hierarchical components of the model
prior = SharedComponent(prior, nLevels)
noiseModel = SharedComponent(noiseModel, nLevels)
vanillaLikelihood = Hierarchy(vanillaLikelihood)
aemLikelihoods = Hierarchy(aemLikelihood)

# build models
surModel = BayesianRegressionModel(vanillaSurLikelihood, prior)
vanillaModel = BayesianModelHierarchy(vanillaLikelihood, prior)
aemModel = BayesianModelHierarchy(aemLikelihood, prior)


# -----------------------------------------------------------------------------
#                             MCMC SETUP
# -----------------------------------------------------------------------------

# Surrogate MRW
# -------------

proposalMVar = 0.15
proposalCov = IIDCovarianceMatrix(DIM, proposalMVar)

mrwBuilder = MRWBuilder()

mrwBuilder.proposalCov = proposalCov
mrwBuilder.bayesModel = surModel

surMRW = mrwBuilder.build_method()


# Burn-In Chain
# -------------

burninBuilder = MLDABuilder()

burninBuilder.basePropCov = proposalCov
burninBuilder.subChainLengths = [20]
burninBuilder.bayesModel = vanillaModel

mldaBurnin = burninBuilder.build_method()


# vanilla MLDA
# ------------

vanillaMLDABuilder = MLDABuilder()

vanillaMLDABuilder.gc.basePropCov = proposalCov
vanillaMLDABuilder.gc.subChainLengths = [6]
vanillaMLDABuilder.gc.bayesModel = vanillaModel

vanillaMLDA = vanillaMLDABuilder.gc.build_method()


# adaptive error correction MLDA
# ------------------------------

aemMLDABuilder = AdaptiveErrorCorrectionMLDABuilder()

aemMLDABuilder.basePropCov = proposalCov
aemMLDABuilder.subChainLengths = [6]
aemMLDABuilder.bayesModel = aemModel

aemMLDA = aemMLDABuilder.build_method()


# -----------------------------------------------------------------------------
#                                INFERENCE
# -----------------------------------------------------------------------------

nSteps = 50000
burnin = 1000

initState = ParameterVector(np.zeros(DIM))

print(f"Running {nSteps} steps of the surrogate MRW")

surMRW.run(nSteps, initState)
states = surMRW.chain.trajectory

thinning = integrated_autocorrelation(states, 'mean')

surMean = np.mean(states[burnin::thinning], axis=0)

# start the actual burn-in chain where the surrogate chain left off
burnin = states[-1]
nSteps = 5000

print(f"\nBurning-in MLDA with {nSteps} steps, starting at last surrogate chain "
      "position")

mldaBurnin.run(nSteps, initState)

initState = mldaBurnin.chain.trajectory[-1]
firstInitStateCopy = np.copy(initState)
mldaBurnin.clear()

nSteps = 50000

print(f"\nRunning {nSteps} steps of vanilla MLDA")

vanillaMLDA.run(nSteps, initState)

print(f"\nRunning {nSteps} of MLDA with an adaptive error model")

assert initState == firstInitStateCopy

aemMLDA.run(nSteps, initState)

posteriorMeanEstimates = []

for mcmc, name in [(vanillaMLDA, "vanilla"),
                   (aemMLDA, "AEM")]:

    states = mcmc.chain.trajectory
    thinning = integrated_autocorrelation(states, 'max')

    posteriorMean = np.mean(states[::thinning], axis=0)
    posteriorMeanEstimates.append(posteriorMean)

    print(f"\n\nResults for {name} MLDA:")
    print("------------------")
    print(f"acceptance rate: {mcmc.diagnostics.global_acceptance_rate()}")
    print(f"IAT estimate: {thinning}")
    print(f"posterior mean: {posteriorMean}")

# -----------------------------------------------------------------------------
#                                PLOTTING
# -----------------------------------------------------------------------------


fig, ax = plt.subplots(1, 3, figsize=(12, 6))

xGrid = np.linspace(-8., 8., 400)
yGrid = np.linspace(-8., 8., 400)

# Create a grid for the contour plot
X, Y = np.meshgrid(xGrid, yGrid)
mesh = np.dstack((X, Y))

# # FIXME: Adjust accordingly from here
# for i in [0, 2]:
#
#     tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
#     surrDensityEval = surrDensity.evaluate_on_mesh(mesh)
#
#     # Plot the contour lines of the target distributions
#     ax[i].contour(
#         X,
#         Y,
#         tgtDensityEval,
#         levels=5,
#         cmap='Reds',
#         label="Target Density")
#     ax[i].contour(
#         X,
#         Y,
#         surrDensityEval,
#         levels=5,
#         cmap='Blues',
#         label="Surrogate Density")
#
#     # Extract x and y coordinates
#     if i == 0:
#         states = vanillaMLDAStates
#         samples = vanillaMLDASamples
#         titleStr = "MLDA"
#
#     else:
#         states = correctedMLDAStates
#         samples = correctedMLDASamples
#         titleStr = "Adaptive MLDA"
#
#         # show shifted surrogate
#         shiftedSurrMean = ParameterVector(
#             surrMean.coefficient - correctedMLDA.surrogate(0).target.correction)
#         shiftedSurrogateDensity = GaussianTargetDensity2d(
#             shiftedSurrMean, surrCov)
#
#         shiftedEval = shiftedSurrogateDensity.evaluate_on_mesh(mesh)
#         ax[1].contour(
#             X,
#             Y,
#             shiftedEval,
#             levels=5,
#             cmap='Greens',
#             label='Shifted Surrogate Density')
#
#         ax[1].annotate("",
#                        xy=(shiftedSurrMean.coefficient[0],
#                            shiftedSurrMean.coefficient[1]),
#                        xytext=(
#                            surrMean.coefficient[0],
#                            surrMean.coefficient[1]),
#                        arrowprops=dict(
#                            arrowstyle="->",
#                            color="black",
#                            lw=3)
#                        )
#
#     meanEst = np.mean(samples, axis=0)
#
#     chainX = [state[0] for state in states]
#     chainY = [state[1] for state in states]
#
#     mcmcX = [sample[0] for sample in samples]
#     mcmcY = [sample[1] for sample in samples]
#
#     # Plot the Markov chain trajectory
#     ax[i].set_title(titleStr)
#     ax[i].set_xlabel('X')
#     ax[i].set_ylabel('Y')
#     ax[i].legend()
#     ax[i].grid(
#         True,
#         which='both',
#         linestyle='--',
#         linewidth=0.5,
#         color='gray',
#         alpha=0.7)
#
#     ax[i].scatter(
#         mcmcX,
#         mcmcY,
#         color='gray',
#         marker='o',
#         s=40,
#         alpha=0.1,
#         label='Selected Samples')
#     ax[i].scatter(
#         meanEst[0],
#         meanEst[1],
#         color='black',
#         s=120,
#         marker='P',
#         label='MCMC Mean Estimate')
#
#
# plt.tight_layout()
# plt.show()
