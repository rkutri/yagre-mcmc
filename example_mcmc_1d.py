import numpy as np
import matplotlib.pyplot as plt
import yagremcmc.postprocessing.autocorrelation as ac

from yagremcmc.test.testSetup import GaussianTargetDensity1d
from yagremcmc.statistics.covariance import IIDCovarianceMatrix
from yagremcmc.chain.method.mrw import MetropolisedRandomWalk
from yagremcmc.chain.diagnostics import AcceptanceRateDiagnostics
from yagremcmc.parameter.scalar import ScalarParameter


tgtMean = ScalarParameter(np.array([1.5]))
tgtVar = 1.
tgtDensity = GaussianTargetDensity1d(tgtMean, tgtVar)

mesh = np.linspace(-5., 5., 200, endpoint=True)

# evaluate target density and normalise
tgtDensityEval = tgtDensity.evaluate_on_mesh(mesh)
tgtDensityEval /= np.sqrt(2. * np.pi * tgtVar)

proposalVariance = 1.5
proposalCov = IIDCovarianceMatrix(1, proposalVariance)

diagnostics = AcceptanceRateDiagnostics()
mcmc = MetropolisedRandomWalk(tgtDensity, proposalCov, diagnostics)

nSteps = int(1e6)
initState = ScalarParameter(np.array([-3.]))
mcmc.run(nSteps, initState)

states = np.array(mcmc.chain.trajectory)

# postprocessing
burnin = int(0.02 * nSteps)

assert nSteps > burnin                                                          
                                                                                
# estimate autocorrelation function                                             
acf = ac.estimate_autocorrelation_function_1d(states[burnin:])
                                                                                
meanIAT = ac.integrated_autocorrelation_nd(states[burnin:], 'mean')             
maxIAT = ac.integrated_autocorrelation_nd(states[burnin:], 'max')               
                                                                                
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

plt.hist(states, bins=50, edgecolor='white', alpha=0.4,
         color='red', density=True, label='mc states')
plt.hist(mcmcSamples, bins=50, edgecolor='black', alpha=0.8,
         color='blue', density=True, label='thinned states')
plt.plot(mesh, tgtDensityEval, color='red', label='target density')
plt.legend()

plt.show()
