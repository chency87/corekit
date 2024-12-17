# corekit
Core kits for sub-projects

## Modules
1. **Context**
Global Context information
- timestamp
- config logging

1. **log**

- log records to logs/log.log. We can change it in the Context
- log records to individual file. use as logger.info(YOU MESSAGE, extra = {'to': 'target path'})

2. **db_manager**
- we could use this to connect to different databases via SQLalchemy, just reminder to update database backends in the function ```python _ensure_connection_string()```. Currently, support the following:
```python
'sqlite' : lambda : URL.create("sqlite", database= os.path.join(host_or_path, database)),
'mysql':  lambda : URL.create("mysql+mysqldb", username= username, password = password, host = host_or_path, port = port, database= database) 
```

- sample small database instance
