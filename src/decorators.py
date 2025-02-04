from threading import Lock

def singleton(cls):
    instance = {}
    _lock: Lock = Lock()
    def _singleton(*args, **kwargs):
        with _lock:
            if cls not in instance:
                instance[cls] = cls(*args, **kwargs)
            return instance[cls]
    return _singleton

class singletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}
    _lock: Lock = Lock()
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]