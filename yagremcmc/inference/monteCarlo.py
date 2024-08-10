# TODO: This is the main class for inference. It acts as the creator object 
#       for specific markov chains as per the factory method pattern.
from yagremcmc.inference.builder import MetropolisHastingsDirector

class MonteCarlo:

    def __init__(self, bayesModel, chainConfig):

        chainDirector = MetropolisHastingsDirector(chainConfig)
        
        self.chain_ = chainDirector.build_chain(bayesModel)


    @property
    def chain(self):
        return self.chain_


    def run(self, nSteps, initialState, verbose=True):
        pass
