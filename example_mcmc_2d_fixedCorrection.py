import numpy as np
import matplotlib.pyplot as plt

from yagremcmc.test.testSetup import GaussianTargetDensity2d
from yagremcmc.parameter.vector import ParameterVector
from yagremcmc.statistics.covariance import (DenseCovarianceMatrix,
                                             IIDCovarianceMatrix)
from yagremcmc.chain.diagnostics import FullDiagnostics
from yagremcmc.chain.method.mlda import MLDABuilder
from yagremcmc.postprocessing.autocorrelation import integrated_autocorrelation


# set up target and surrogate measures. Surrogate measure concentrates on a
# larger region and is shifted
tgtMean = ParameterVector(np.array([0.0, 0.5]))
tgtCov = np.array([[1.1, -0.4], [-0.4, 0.6]])

tgtDensity = GaussianTargetDensity2d(tgtMean, tgtCov)

meanShift = np.array([1.8, -2.4])
surrMean = ParameterVector(tgtMean.coefficient + meanShift)
surrCov = np.array([[1.6, 0.2], [0.2, 1.1]])

surrDensity = GaussianTargetDensity2d(surrMean, surrCov)

# set MLDA-specific parameters
basePropVar = 0.75
baseProposalCov = IIDCovarianceMatrix(tgtMean.dimension, basePropVar)
subChainLength = 6

# set up builders for vanilla and corrected MLDA variants
builder = MLDABuilder()

builder.explicitTarget = tgtDensity
builder.surrogateTargets = [surrDensity]
builder.baseProposalCovariance = baseProposalCov
builder.subChainLengths = [4]

vanillaMLDA = builder.build_method()

builder.biasCorrection = [meanShift]
correctedMLDA = builder.build_method()

# run parameters
nSteps = 50000
initState = ParameterVector(np.array([-9., -7.]))

# run both chains
vanillaMLDA.run(nSteps, initState)
vanillaMLDAStates = np.array(vanillaMLDA.chain.trajectory)

correctedMLDA.run(nSteps, initState)
correctedMLDAStates = np.array(correctedMLDA.chain.trajectory)

# postprocessing
dim = tgtMean.dimension
burnIn = 100

assert nSteps > burnIn

vanillaMLDAThinning = integrated_autocorrelation(
    vanillaMLDAStates[burnIn:], 'max')
vanillaMLDASamples = vanillaMLDAStates[burnIn::vanillaMLDAThinning]

correctedMLDAThinning = integrated_autocorrelation(
    correctedMLDAStates[burnIn:], 'max')
correctedMLDASamples = correctedMLDAStates[burnIn::correctedMLDAThinning]

# compare diagnostics
print("\nMLDA Analytics")
print("--------------")
print(f"acceptance rate: {vanillaMLDA.diagnostics.global_acceptance_rate()}")
print(f"IAT estimate: {vanillaMLDAThinning}")

print("\nAdaptive MLDA Analytics")
print("-----------------------")
print(f"acceptance rate: {correctedMLDA.diagnostics.global_acceptance_rate()}")
print(f"IAT estimate: {correctedMLDAThinning}")
print(f"error correction: {correctedMLDA.surrogate(0).target.correction}")

print("\nMLDA Inference")
print("--------------")
print(f"true mean: {tgtMean.coefficient}")
print(f"mean state: {np.mean(vanillaMLDAStates, axis=0)}")
print(f"mean estimate: {np.mean(vanillaMLDASamples, axis=0)}")

print("\nAdaptive MLDA Inference")
print("--------------")
print(f"true mean: {tgtMean.coefficient}")
print(f"mean state: {np.mean(correctedMLDAStates, axis=0)}")
print(f"mean estimate: {np.mean(correctedMLDASamples, axis=0)}")
print("\n\n\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

xGrid = np.linspace(-8., 8., 400)
yGrid = np.linspace(-8., 8., 400)

# Create a grid for the contour plot
X, Y = np.meshgrid(xGrid, yGrid)
mesh = np.dstack((X, Y))

for i in [0, 1]:
    tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
    surrDensityEval = surrDensity.evaluate_on_mesh(mesh)

    # Plot the contour lines of the target distributions
    ax[i].contour(
        X,
        Y,
        tgtDensityEval,
        levels=5,
        cmap='Reds',
        label="Target Density")
    ax[i].contour(
        X,
        Y,
        surrDensityEval,
        levels=5,
        cmap='Blues',
        label="Surrogate Density")

    # Extract x and y coordinates
    if i == 0:
        states = vanillaMLDAStates
        samples = vanillaMLDASamples
        titleStr = "MLDA"

    else:
        states = correctedMLDAStates
        samples = correctedMLDASamples
        titleStr = "Adaptive MLDA"

        # show shifted surrogate
        shiftedSurrMean = ParameterVector(
            surrMean.coefficient - correctedMLDA.surrogate(0).target.correction)
        shiftedSurrogateDensity = GaussianTargetDensity2d(
            shiftedSurrMean, surrCov)

        shiftedEval = shiftedSurrogateDensity.evaluate_on_mesh(mesh)
        ax[1].contour(
            X,
            Y,
            shiftedEval,
            levels=5,
            cmap='Greens',
            label='Shifted Surrogate Density')

        ax[1].annotate("",
                       xy=(shiftedSurrMean.coefficient[0],
                           shiftedSurrMean.coefficient[1]),
                       xytext=(
                           surrMean.coefficient[0],
                           surrMean.coefficient[1]),
                       arrowprops=dict(
                           arrowstyle="->",
                           color="black",
                           lw=3)
                       )

    meanEst = np.mean(samples, axis=0)

    chainX = [state[0] for state in states]
    chainY = [state[1] for state in states]

    mcmcX = [sample[0] for sample in samples]
    mcmcY = [sample[1] for sample in samples]

    # Plot the Markov chain trajectory
    ax[i].set_title(titleStr)
    ax[i].set_xlabel('X')
    ax[i].set_ylabel('Y')
    ax[i].legend()
    ax[i].grid(
        True,
        which='both',
        linestyle='--',
        linewidth=0.5,
        color='gray',
        alpha=0.7)

    ax[i].scatter(
        mcmcX,
        mcmcY,
        color='gray',
        marker='o',
        s=40,
        alpha=0.1,
        label='Selected Samples')
    ax[i].scatter(
        meanEst[0],
        meanEst[1],
        color='black',
        s=120,
        marker='P',
        label='MCMC Mean Estimate')


# Show the plot after the loop
plt.tight_layout()
plt.show()
