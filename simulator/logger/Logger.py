#!/usr/bin/env python
""" 
Author: Giorgio Raineri
"""

# Import standard modules
import sys
import os

# Import Libraries
from loguru import logger

# Import Application Modules

FOLDER_NAME = 'logs'
FOLDER_PATH = f'/{FOLDER_NAME}'
# Initialize logger
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <blue>{module}</blue> | " \
                "<level>{level}</level> |  <level>{message}</level> "
logger.remove()

os.makedirs(FOLDER_PATH, exist_ok=True)
# Create a logfile when the previous one reaches 2MB in size, keep the last 10 logfiles
logger.add(FOLDER_PATH + '/out_{time:DD-MM-YYYY}.log', format=logger_format, level='DEBUG', rotation="2 MB", retention=10, backtrace=False) # General logfile
logger.add(FOLDER_PATH + '/error_{time:DD-MM-YYYY}.log', format=logger_format, level='WARNING', rotation="2 MB", retention=10, backtrace=True, diagnose=True) # Error logfile
