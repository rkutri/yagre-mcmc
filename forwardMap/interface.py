from abc import ABC, abstractproperty, abstractmethod

from parameter.interface import ParameterInterface


class ForwardMapInterface(ABC):                                                 
    """
    Common interface for the forward map in the inverse problem. If an external
    library is used for the model evaluation, this serves as the base for
    the implementation of a corresponding proxy class.
    """
                                                                                
    @abstractproperty                                                           
    def parameter(self):                                                        
        pass                                                                    
                                                                                
    @parameter.setter                                                           
    def parameter(self, parameter: ParameterInterface) -> None:                 
        pass                                                                    
                                                                                
    @abstractmethod                                                             
    def evaluate(self, x):                                                      
        pass        


class EvaluationRequest(ABC):
    pass
