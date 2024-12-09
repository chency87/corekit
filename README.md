# corekit
Core kits for sub-projects

## Modules
1. **log**

- log records to logs/log.log. We can change it in the src.log file
- log records to individual file. we could specify different single file in log_config.ini. follow the instructions of log_config.ini
- predefined metrics/agent tags. we could log records by logging.getLogger('metrics.query0'). this will create a metrics_query0.log file under folder logs/

2. **db_manager**
- we could use this to connect to different databases via SQLalchemy, just reminder to update database backends in the function ```python _ensure_connection_string()```. Currently, support the following:
```python
'sqlite' : lambda : URL.create("sqlite", database= os.path.join(host_or_path, database)),
'mysql':  lambda : URL.create("mysql+mysqldb", username= username, password = password, host = host_or_path, port = port, database= database) 
```
