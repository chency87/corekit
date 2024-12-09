import sys
import os
# Get the current directory (where your_script.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from src import *
import random, logging

logger = logging.getLogger('src.query')
metrics_logger =  logging.getLogger('metrics.query')

agent_logger =  logging.getLogger('agent.worker1')

# print(type(logger))

logger.info('this is a test of log')
logger.info('this is a test of log')
logger.error('this is a test of log')
logger.warning('this is a test of log')

metrics_logger.info('this is a metrics')

agent_logger.info('message from agent')