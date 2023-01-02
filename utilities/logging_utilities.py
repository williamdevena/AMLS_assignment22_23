import logging


def print_name_stage_project(stage):
    '''
    At the beginning of a stage (like Data preparation, data cleaning, ...) it prints
    the name of the stage.

    Args:
        - stage (str): name of the stage

    Returns:
        - None

    '''
    width = 50+len(stage)
    logging.info("-"*(width))
    logging.info("-"*(width))
    logging.info(f"-----------------------  {stage}  -----------------------")
    logging.info("-"*(width))
    logging.info(("-"*(width))+"\n\n")
