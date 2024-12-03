import logging


def create_logger(name=None):
    """
    Create a logger with a unique name.
    If no name is provided, a unique logger will be created using id().
    """
    if name is None:
        name = f"defaultLogger_{id(name)}"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create a console handler
        consoleHandler = logging.StreamHandler()

        # Set a logging format
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        consoleHandler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(consoleHandler)

    return logger
