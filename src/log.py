import logging
import logging.config as logconfig
import logging.config
from logging.handlers import QueueListener
import sys, atexit, asyncio, threading
import multiprocessing
from datetime import datetime
from typing import Optional, List, Union
from pathlib import Path
logging.config.fileConfig
LOGGING_FILE_PREFIX = datetime.now().strftime('%Y-%m-%d_%H-%M') #_%H-%M-%S _%H_%M
GLOBAL_LOGGER_HANDLERS: List[logging.Handler] = []

'''
    Use QueueListener to implement async log modules 
'''

def assert_folder(fp):
    if not Path(fp).exists():
        Path(fp).mkdir(parents= True, exist_ok= True)
    
    return fp


DEFAULT_LOG_FILE_PATH = 'logs/log.log'
DEFAULT_INDIVIDUAL_FILE_PATH = 'logs/individual.log'

def _strip_spaces(alist):
    return map(str.strip, alist)

class AsyncQueueListener(QueueListener):
    loop: Optional[asyncio.AbstractEventLoop] = None
    def __init__(self, queue, *handlers: logging.Handler, respect_handler_level: bool = False) -> None:
        super().__init__(queue, *handlers, respect_handler_level=respect_handler_level)

    @classmethod
    def _start(cls):
        if cls.loop is None:            
            cls.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.loop)
    def _monitor(self) -> None:
        super()._monitor()

    def stop(self) -> None:
        if self.loop is not None:
            self.loop.stop()
            self.loop.close()
        self.enqueue_sentinel()
        if self._thread:
            self._thread.join(1)
            self._thread = None


class AppLogger(logging.Logger):
    def __init__(self, name, level = 0):
        logging.Logger.__init__(self, name, level)
        conf = self.fileconfig()
        logger_keys = conf['loggers']['keys'].split(',')
        name_prefix = name.split('.')[0]
        if not self.handlers and name_prefix in logger_keys:
            LOG_FOLDER =  assert_folder(f'logs/{LOGGING_FILE_PREFIX}')
            self.setup_logger(conf, name,  LOG_FOLDER = LOG_FOLDER) 
    def setup_logger(self, cp, name, LOG_FOLDER = 'logs'):

        section_name = name.split('.')[0]
        section_name = 'logger_%s' % section_name
        if not cp.has_section(section_name):
            return
        section = cp[section_name]
        log_file_fp = LOG_FOLDER + '/' +'log.log'
        if section.getint("singlefile",  fallback=0):
            log_file_fp = LOG_FOLDER + '/' + '_'.join(name.split('.')) + '.log'
        if 'level' in section:
            level = section["level"]
            self.setLevel(level)
        propagate = section.getint("propagate", fallback=1)
        self.propagate = propagate
        hlist = section["handlers"]
        if len(hlist):
            hlist = hlist.split(",")
            hlist = _strip_spaces(hlist)
            formatters = self._get_formatter(cp)
            for hand in hlist:
                self._setup_handler(cp, hand, formatters, LOG_FILE_PATH = log_file_fp )

    def fileconfig(self):
        from configparser import ConfigParser
        conf = ConfigParser()
        try:
            config_fp = Path(__file__).parent.parent / 'log_config.ini'
            conf.read(str(config_fp.absolute()), encoding='utf-8')
        except Exception as e:
            ...
        return conf

    def _get_formatter(self, cp):
        flist = cp["formatters"]["keys"]
        if not len(flist):
            return {}
        flist = flist.split(",")
        flist = _strip_spaces(flist)
        formatters = {}
        for form in flist:
            sectname = "formatter_%s" % form
            fs = cp.get(sectname, "format", raw=True, fallback=None)
            dfs = cp.get(sectname, "datefmt", raw=True, fallback=None)
            stl = cp.get(sectname, "style", raw=True, fallback='%')
            c = logging.Formatter
            f = c(fs, dfs, stl)
            formatters[form] = f
        return formatters
    def _setup_handler(self, cp, handler_name, formatters, **kwargs):
        """Install and return handlers"""
        section = cp["handler_%s" % handler_name]
        klass = section["class"]
        fmt = section.get("formatter", "")
        try:
            klass = eval(klass, vars(logging))
        except (AttributeError, NameError):
            raise NameError(f'could not initialize logger handler named: {handler_name}')
        args = section.get("args", '()')
        if '%LOG_FILE_PATH%' in args:
            args = args.replace('%LOG_FILE_PATH%', kwargs.get('LOG_FILE_PATH', DEFAULT_LOG_FILE_PATH))
        if 'LOG_INDIVIDUAL_PATH' in args:
            args = args.replace('%LOG_INDIVIDUAL_PATH%', kwargs.get('LOG_FILE_PATH', DEFAULT_INDIVIDUAL_FILE_PATH))
        args = eval(args, vars(logging))
        kwargs = section.get("kwargs", '{}')
        kwargs = eval(kwargs, vars(logging))
        h = klass(*args, **kwargs)
        if "level" in section:
            level = section["level"]
            h.setLevel(level)
        if len(fmt):
            h.setFormatter(formatters[fmt])
        
        if section.get('queue', fallback = 0) and isinstance(h, logging.FileHandler):
            queue = multiprocessing.Queue()
            listener = AsyncQueueListener(queue, h)
            self.addHandler(logging.handlers.QueueHandler(queue))
            GLOBAL_LOGGER_HANDLERS.append((listener, self))
            listener.start()
            atexit.unregister(_close_loggers)
            atexit.register(_close_loggers)
        else:
            self.addHandler(h)

    def setup_metrics_logger(self, cp, name: str, LOG_FILE_PATH):
        name = name[8:]
        section_name = f'logger_%s' % section_name
        if cp.has_section(section_name):
            section = cp[section_name]
            if 'level' in section:
                level = section["level"]
                self.setLevel(level)
            propagate = section.getint("propagate", fallback=1)
            self.propagate = propagate


        if not self.handlers:
            target_folder = f'results/{LOGGING_FILE_PREFIX}'
            assert_folder(target_folder)
            file_handler = logging.FileHandler(filename = f'{target_folder}/{name}.log', mode = 'w')
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(10)
            queue = multiprocessing.Queue()
            listener = AsyncQueueListener(queue, file_handler)
            self.addHandler(logging.handlers.QueueHandler(queue))
            GLOBAL_LOGGER_HANDLERS.append((listener, self))
            listener.start()
            atexit.unregister(_close_loggers)
            atexit.register(_close_loggers)

logging.setLoggerClass(AppLogger)

def _close_loggers():
    while GLOBAL_LOGGER_HANDLERS:
        listener, logger = GLOBAL_LOGGER_HANDLERS.pop()
        logger.handlers = listener.handlers
        listener.stop()
