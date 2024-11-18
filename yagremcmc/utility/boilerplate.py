import logging

def create_logger():

    logger = logging.getLogger(__name__)                                          
    logger.setLevel(logging.INFO)                                                 
										    
    # Create a console handler                                                      
    consoleHandler = logging.StreamHandler()                                        
										       
    # Set a logging format                                                             
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s: %(message)s')                     
    consoleHandler.setFormatter(formatter)                                          
										    
    # Add the console handler to the mhLogger                                       
    logger.addHandler(consoleHandler) 

    return logger
