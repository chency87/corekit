import sys
import os
# Get the current directory (where your_script.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from src.context import get_ctx
import logging 

c = get_ctx()
logger = logging.getLogger('src.test')
logger.info('this is a test log', extra = {})
logger.warning('this is a warning message', extra = {'to': 'extra'})
logger.warning('this is a warning message', extra = {'to': 'extra2'})
logger.warning('this is a warning message', extra = {'to2': 'extra2'})
logger.critical('this is a critical message')
