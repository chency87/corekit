


from typing import List, Optional, Dict, NewType
from logging.handlers import QueueListener
import logging, os, sys
from threading import Lock

Unique_Id = NewType('Unique_Id', str)

def get_console_handler(level, formatter):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    return console_handler

def get_file_handler(fp, level, formatter):
    file_handler = logging.handlers.TimedRotatingFileHandler(fp, when='D', interval= 1, delay= True)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    return file_handler


import ast, json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg if isinstance(record.msg, dict) else record.getMessage()
        try:
            msg = ast.literal_eval(message)
            message = json.dumps(msg)
        finally:
            return message

class TelemetryListener(QueueListener):
    GLOBAL_HANDLERS: Dict[str, logging.Handler] = {}
    '''
        Use logging QueueListener to process all metrics related record.
        use as:
            logger = logging.getLogger()
            logger.info('db_id: 5, eq:0', extra= {'to': /path/to/save})
            logger.info('db_id: 5, eq:0', extra= {'to': /path/to/save})
    '''
    def __init__(self, queue, save_path):
        super().__init__(queue, respect_handler_level=False)
        self.save_path = save_path
        self._lock = Lock()

    def ensure_handler(self, hld_name, file_fp):
        with self._lock:
            if hld_name not in self.GLOBAL_HANDLERS:
                handler = logging.FileHandler(filename= file_fp)
                handler.setLevel('DEBUG')
                handler.setFormatter(JsonFormatter())
                self.GLOBAL_HANDLERS[hld_name] = handler
        return self.GLOBAL_HANDLERS[hld_name]

    def handle(self, record):
        """
            Handle a record.
            This just loops through the handlers offering them the record to handle.
        """
        record = self.prepare(record)
        if isinstance(record, logging.LogRecord) and 'to' in record.__dict__:
            to_ = getattr(record, 'to')
            hld_fp = os.path.join(self.save_path, f'{to_}.log')
            handler = self.ensure_handler(to_, hld_fp)
            handler.handle(record)

        

# import logging

# KEEP = 'keep'

# def add_logging_level(level_name: str, level_num: int, method_name=None,
#                       if_exists=KEEP, *, exc_info=False, stack_info=False):
#     """
#     Comprehensively adds a new logging level to the `logging` module and the
#     currently configured logging class.

#     `levelName` becomes an attribute of the `logging` module with the value
#     `levelNum`. `method_name` becomes a convenience method for both `logging`
#     itself and the class returned by `logging.getLoggerClass()` (usually just
#     `logging.Logger`). If `method_name` is not specified, `levelName.lower()` is
#     used.

#     To avoid accidental clobberings of existing attributes, this method will
#     raise an `AttributeError` if the level name is already an attribute of the
#     `logging` module or if the method name is already present 

#     Example
#     -------
#     >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
#     >>> logging.getLogger(__name__).setLevel("TRACE")
#     >>> logging.getLogger(__name__).trace('that worked')
#     >>> logging.trace('so did this')
#     >>> logging.TRACE
#     5

#     """
#     if not method_name:
#         method_name = level_name.lower()

#     if hasattr(logging, level_name):
#        raise AttributeError('{} already defined in logging module'.format(level_name))
#     if hasattr(logging, method_name):
#        raise AttributeError('{} already defined in logging module'.format(method_name))
#     if hasattr(logging.getLoggerClass(), method_name):
#        raise AttributeError('{} already defined in logger class'.format(method_name))

#     def for_logger_adapter(self, msg, *args, **kwargs):
#         kwargs.setdefault('exc_info', exc_info)
#         kwargs.setdefault('stack_info', stack_info)
#         kwargs.setdefault('stacklevel', 2)
#         self.log(level_num, msg, *args, **kwargs)

#     def for_logger_class(self, msg, *args, **kwargs):
#         if self.isEnabledFor(level_num):
#             kwargs.setdefault('exc_info', exc_info)
#             kwargs.setdefault('stack_info', stack_info)
#             kwargs.setdefault('stacklevel', 2)
#             self._log(level_num, msg, args, **kwargs)

#     def for_logging_module(*args, **kwargs):
#         kwargs.setdefault('exc_info', exc_info)
#         kwargs.setdefault('stack_info', stack_info)
#         kwargs.setdefault('stacklevel', 2)
#         logging.log(level_num, *args, **kwargs)

    
#     def logForLevel(self, message, *args, **kwargs):
#         if self.isEnabledFor(level_num):
#             self._log(level_num, message, args, **kwargs)
#     def logToRoot(message, *args, **kwargs):
#         logging.log(level_num, message, *args, **kwargs)

#     logging.addLevelName(level_num, level_name)
#     setattr(logging, level_name, level_num)
#     setattr(logging.getLoggerClass(), method_name, logForLevel)
#     setattr(logging, method_name, logToRoot)




# # import logging.handlers
# # import logging, queue
# # import atexit
# # from typing import Dict, NewType
# # from pathlib import Path
# # from . import get_ctx

# # # # from .context import get_ctx
# # # import logging.handlers
# # # from typing import List
# # # GLOBAL_LOGGER_HANDLERS: List[logging.Handler] = []


# # ## 1. set logger level

# # ## 2. add queue handler 

# # ## 3. implement a queue listener

# # PROJECT_PATH = Path(__file__).resolve().parent.parent

# # print(__file__)
# # print(__name__)

# # print(PROJECT_PATH.name)


# # HANDLER_NAME = NewType('HANDLER_NAME', str)
# # GLOBAL_HANDLERS: Dict[HANDLER_NAME, logging.Handler] = {}
# # APP_ROOT_LOGGER = None

# # import logging
# # import sys
# # from logging.handlers import TimedRotatingFileHandler
# # FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

# # def get_console_handler(level = logging.DEBUG):
# #    console_handler = logging.StreamHandler(sys.stdout)
# #    console_handler.setFormatter(FORMATTER)
# #    console_handler.setLevel(level)
# #    return console_handler

# # def get_file_handler(LOG_FILE):
# #    file_handler = TimedRotatingFileHandler(LOG_FILE, when='D', interval= 1, delay= True)
# #    file_handler.setFormatter(FORMATTER)
# #    return file_handler

# # def get_logger(logger_name):
# #    logger = logging.getLogger(logger_name)
# #    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
# #    logger.addHandler(get_console_handler())
# #    logger.addHandler(get_file_handler())
# #    # with this pattern, it's rarely necessary to propagate the error up to parent
# #    return logger



# # def setup_logger(root = __name__.split('.')[0]):
# #     print(f'setup looger {root}')
# #     corekit_logger = logging.getLogger(root)
# #     # Set the logger's log level
# #     corekit_logger.setLevel(logging.DEBUG)
# #     # Set console handler
# #     console_handler = logging.StreamHandler()
# #     console_handler.setLevel(logging.DEBUG)
# #     console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)d - %(message)s", datefmt= "%Y-%m-%d %H:%M:%S")
# #     console_handler.setFormatter(console_formatter)
# #     corekit_logger.addHandler(console_handler)
# #     corekit_logger.addHandler(logging.StreamHandler())

    


# # setup_logger()




# # import sys
# # class AppLogger(logging.Logger):
# #     def __init__(self, name, level = 0):
# #         logging.Logger.__init__(self, name, level)
# #         name_prefix = name.split('.')[0]

# #         if not self.ensure_handler():
# #             ...

# #     def ensure_handler(self):
# #         name = self.name
# #         i = name.rfind(".")
# #         rv = None
# #         while (i > 0) and not rv:
# #             substr = name[:i]
# #             if substr in self.manager.loggerDict:
# #                 alogger = self.manager.loggerDict[substr]
# #                 if isinstance(alogger, logging.Logger) and alogger.hasHandlers():
# #                     return True
# #             i = name.rfind(".", 0, i - 1)        
# #         return False

# # #     def telemetry(self, msg, *args, exc_info = None, stack_info = False, stacklevel = 1, extra = None):        
# # #         return self.info(msg, *args, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
    
# # logging.setLoggerClass(AppLogger)
# # logger = logging.getLogger('corekit')
# # logger.addHandler(logging.StreamHandler())


# # cklogger = logging.getLogger('corekit.src')

# # logger1 = logging.getLogger('src.app.name1')
# # logger2 = logging.getLogger('src.app.name1.l2')
# # logger3 = logging.getLogger('src.app.name1.l3')
# # logger4 = logging.getLogger('src.app3')


# # print(f'{type(cklogger.parent)} --> {cklogger.parent.name}')
# # print(f'{type(logger.parent)} --> {logger.parent.name}')
# # print(f'{type(logger1.parent)} --> {logger1.parent.name}')


# # # def config_logging():
# # #     ...

# # # # logging.addLevelName( 80, 'agent')

# # # # logging._acquireLock()
# # # # level_name = 'TELEMETRY'
# # # # # registered_num = logging.getLevelName(level_name)
# # # # # logger_class = logging.getLoggerClass()
# # # # # logger_adapter = logging.LoggerAdapter

# # # # # setattr(logging, level_name, level_num)
# # # # setattr(logging.getLoggerClass(), 'telemetry', telemetry)
# # # # method_name = 'telemetry'
# # # # # setattr(logging, method_name, for_logging_module)
# # # # # setattr(logger_class, method_name, for_logger_class)
# # # # # setattr(logger_adapter, method_name, for_logger_adapter)

# # # # logging._releaseLock()

# # # logging.setLoggerClass(AppLogger)

# # # logger = logging.getLogger('aggg')

# # # l2 = logging.getLogger('aggg')

# # # print(logger is l2)

# # # # import sys
# # # # logger.addHandler(logging.StreamHandler(stream = sys.stdout))
# # # # logger.setLevel(logging.INFO)
# # # # logger.telemetry('this is an agent message')

# # # # l2 = logging.getLogger('aggg')
# # # # # logger.manager.getLogger('aggg')
# # # # ha = logging.FileHandler('logg.a', mode= 'w')
# # # # ha.set_name('aa')
# # # # hb = logging.FileHandler('logg.a',  mode= 'w')
# # # # hb.set_name('aa')
# # # # print(ha == hb)
# # # # print(ha is hb)
# # # # print(ha)
# # # # print(hb)
# # # # print(hb in [ha])
# # # # print(ha.get_name())
# # # # print(hb.get_name())

# # # # ha.addFilter()


# # # # class UptimeEndpointFilter(logging.Filter):
# # # #     def filter(self, record: logging.LogRecord) -> bool:
# # # #         # print(record)
# # # #         # print(record.get('agent', ''))
# # # #         print('agent' in record.__dict__)
# # # #         if "GET /up" in record.msg:
# # # #             return False
# # # #         else:
# # # #             return True
# # # # logger.addFilter(UptimeEndpointFilter(name= 'filtera'))


# # # # logger.info('aaaa, ext', extra= {'agent': True})