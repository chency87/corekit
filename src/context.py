
'''
    A Global context manages all execution information, global configuration options, etc
       
        such as app_name, log prefix, uuid of current execution
        

'''
import logging.handlers
from typing import NewType, Dict
from datetime import datetime
from logging.handlers import QueueListener, QueueHandler
from queue import Queue
import logging, sys, atexit, os
from .decorators import singleton
from .io import assert_folder
from .log import TelemetryListener, get_console_handler, get_file_handler

ContextId = NewType('ContextId', str)

DEFAULT = {
    'ctx_id' : datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    'app_name': 'corekit',
    'log_level': 'DEBUG',
    'log_file': 'logs/log.log',
    'result_path': 'logs/',
    'format': '%(asctime)s | %(name)s | %(levelname)8s | %(filename)s line:%(lineno)d | %(message)s',
    'datefmt': '%b %d %H:%M:%S',
    'stage': 'dev'    
}

@singleton
class Context:
    __slots__ = ('ctx_id', 'app_name', 'result_path', 'log_level', 'log_file', 'format', 'datefmt', 'stage',  '_logger', 'q')
    def __init__(self, **kwargs):
        kv = DEFAULT.copy()
        kv.update(kwargs)
        for k, v in kv.items():
            setattr(self, k, v)
        self.result_path = os.path.join(self.result_path, self.ctx_id)
        assert_folder(self.result_path)
        self.q = Queue(-1)
        self.config()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self._config_log()
        return self._logger
    
    def config(self):
        self._logger = self._config_log()
        self._config_telemetry()

    def _config_log(self):
        root = __name__.split('.')[0]
        fmt = logging.Formatter(fmt= self.format, datefmt = self.datefmt)
        print(f'set app root logger to {root}')        
        corekit_logger = logging.getLogger(root)
        corekit_logger.setLevel(self.log_level)
        corekit_logger.addHandler(get_console_handler(self.log_level, fmt))
        corekit_logger.addHandler(get_file_handler(fp = self.log_file, level= self.log_level, formatter= fmt))
        return corekit_logger

    def _config_telemetry(self):
        queue_handler = QueueHandler(self.q)
        class MetricsFilter(logging.Filterer):
            def filter(self, record):
                return 'to' in record.__dict__
        queue_handler.addFilter(MetricsFilter())
        self.logger.addHandler(queue_handler)
        listener = TelemetryListener(queue= self.q, save_path= self.result_path)
        listener.start()
        atexit.register(listener.stop)
    def __repr__(self):
        return f"Context({self.ctx_id})"

# Global context
_main_ctx = None
def get_ctx(**kwargs) -> Context:
    '''
    Return a reference to the global context.
    Args:
        ctx_id, current context 
    '''
    global _main_ctx
    if _main_ctx is None:
        _main_ctx = Context(**kwargs)
    return _main_ctx
