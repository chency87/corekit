from __future__ import annotations
from sqlalchemy import __version__ as SQLALCHEMY_VERSION
from sqlalchemy.schema import CreateTable
from packaging.version import Version
if Version(SQLALCHEMY_VERSION) < Version("1.5"):
    from sqlalchemy.engine import URL, Engine, Connection
    from sqlalchemy import create_engine, text, MetaData
else:
    from sqlalchemy import create_engine, text, Connection, MetaData, Engine, URL

from threading import Lock
from typing import List, Tuple, Any, Dict, Union
from collections import defaultdict
from ..decorators import *
import random

logger = logging.getLogger(__name__)


class Connect:
    def __init__(self, connection: Connection):
        self.conn: Connection = connection
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        self.conn.close()

    def execute(self, stmt, fetch: Union[str, int] = 'all', commit = False):
        r = self.conn.execute(text(stmt))
        results = None
        if fetch == 'all':
            results = r.fetchall()
        elif fetch == 'one' or str(fetch) == '1':
            results = r.fetchone()
        elif fetch == 'random':
            samples = r.fetchmany(10)
            results = random.choice(samples) if samples else []
        elif isinstance(fetch, int):
            results = r.fetchmany(fetch)
        
        if commit:
            self.conn.commit()
        return results
        
    def create_tables(self, *ddls):
        for ddl in ddls:
            self.execute(ddl, fetch= None)

    def drop_table(self, table_name):
        self.execute(f"DROP TABLE IF EXISTS {table_name}", fetch= None)


class DBManager(metaclass = singletonMeta):
    '''
        Maintain a connection pool to connect to various databases. Use as 
        with DBManager().get_connection(host_or_path= host, database= db_name, username= username, password= password, dialect= 'mysql') as conn:
            conn.create_tables(...)
            records = conn.execute(stmt, fetch = 'all')
    '''
    def __init__(self, **kwargs):
        self.engines:Dict[str, Dict[str, Engine]] = defaultdict(dict)
        self.lock = Lock()
        self.set_options('MAX_CHECKOUTS', kwargs.get('max_checkouts', 100))

    def set_options(self, key, value):
        setattr(self, key, value)

    def _ensure_connection_string(self, host_or_path, database, port = None, username = None, password = None, dialect = 'sqlite') -> URL:
        import os
        mapping = {
            'sqlite' : lambda : URL.create("sqlite", database= os.path.join(host_or_path, database)),
            'mysql':  lambda : URL.create("mysql+mysqldb", username= username, password = password, host = host_or_path, port = port, database= database)   
        }
        return mapping[dialect]()

    def _assert_engine(self, conn_str: URL, pool_size=20, max_overflow=10, pool_timeout=15, pool_recycle=60, **kwargs)-> Engine:
        host_or_path = conn_str.host or conn_str.database
        with self.lock:
            if conn_str in self.engines[host_or_path]:
                return self.engines[host_or_path][conn_str]
            
            if self._get_checkouts(host_or_path) > self.MAX_CHECKOUTS:
                self._clean_unused_engine(host_or_path)
            
            if conn_str not in self.engines[host_or_path]:
                engine = create_engine(
                    conn_str,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_recycle=pool_recycle
                )
                self.engines[host_or_path][conn_str] = engine
                logger.info(f'create new connection for {conn_str}')
            return self.engines[host_or_path][conn_str]

    def get_connection(self, host_or_path, database, port = None, username = None, password = None, dialect = 'sqlite',
                       pool_size=20, max_overflow=10, pool_timeout=15, pool_recycle=60, **kwargs) -> Connect:
        conn_str = self._ensure_connection_string(host_or_path, database= database, port= port, username= username, password= password, dialect= dialect)        
        engine = self._assert_engine(conn_str, pool_size, max_overflow, pool_timeout, pool_recycle, **kwargs)
        return Connect(connection= engine.connect())

    def get_schema(self, host_or_path, database, port = None, username = None, password = None, dialect = 'sqlite') -> str:
        conn_str = self._ensure_connection_string(host_or_path, database= database, port= port, username= username, password= password, dialect= dialect)
        engine = self._assert_engine(conn_str)
        metadata = MetaData()
        metadata.reflect(bind = engine)

        schema = []
        for table_name, table in metadata.tables.items():
            ddl = str(CreateTable(table).compile(engine))
            schema.append(ddl)
        return '\n'.join(schema)
        
    def _get_checkouts(self, host_or_path) -> int:
        '''
            Return the count of connections in use
        '''
        availables = self.engines[host_or_path]
        checkouts = 0
        for _, engine in availables.items():
            checkouts = checkouts + engine.pool.size() + engine.pool.overflow()
        return checkouts

    def _clean_unused_engine(self, host_or_path):
        '''
            ensure thread safe to clears engiens
        '''
        unused = 0
        for conn_str in list(self.engines[host_or_path].keys()):
            engine = self.engines[host_or_path][conn_str]
            if engine.pool.checkedout() < 1:
                unused += 1
                engine.dispose()
                del self.engines[host_or_path][conn_str]
        logger.debug(f'Cleaned {unused} unused connections in the connection pool')

    def export_database(self, host_or_path, database, port = None, username = None, password = None, dialect = 'sqlite') -> List[str]:
        database_dump = []

        conn_str = self._ensure_connection_string(host_or_path, database= database, port= port, username= username, password= password, dialect= dialect)
        engine = self._assert_engine(conn_str)
        metadata = MetaData()
        metadata.reflect(bind = engine)
        with engine.connect() as conn:
            for table_name, table in metadata.tables.items():
                database_dump.append(str(CreateTable(table).compile(engine)))
                result = conn.execute(table.select())
                for row in result:
                    row = row._asdict()
                    columns = ", ".join(row.keys())
                    values = ", ".join(
                        f"'{value}'" if value is not None else "NULL" for value in row.values()
                    )
                    insert_stmt = f"INSERT INTO {table_name} ({columns}) VALUES ({values});"
                    database_dump.append(insert_stmt)
        return database_dump
    
    def export_sampledb(self, host_or_path, database, to, port = None, username = None, password = None,
                        query = None, size = 10,  dialect = 'sqlite'):
        
        from sqlglot import parse_one, exp
        must_contain = {}
        if query is not None:
            try:
                expr = parse_one(sql = str(query), dialect = dialect)
                for pred in expr.find_all(exp.Predicate):
                    tables = set()
                    for col in pred.find_all(exp.Column):
                        tables.add(col.table)
                    if len(tables) > 0 and len(tables) < 2:
                        tbl = tables.pop()
                        must_contain[tbl] = exp.select('*').from_(tbl).where(pred)
            except Exception as e:
                ...
        
        database_dump = []
        schema_dump = []
        conn_str = self._ensure_connection_string(host_or_path, database= database, port= port, username= username, password= password, dialect= dialect)
        engine = self._assert_engine(conn_str)
        metadata = MetaData()
        metadata.reflect(bind = engine)

        with engine.connect() as conn:
            for table_name, table in metadata.tables.items():
                schema_dump.append(str(CreateTable(table).compile(engine)))
                records_dump = []

                pks_values = {}
                for col_name, column in table.columns.items():
                    if column.primary_key:
                        pks_values[col_name] = []

                stmt = table.select()
                if table_name in must_contain:
                    stmt = text(must_contain[table_name])
                result = conn.execute(stmt)
                for row in result:
                    row = row._asdict()
                    columns = ", ".join([f"`{k}`" for k in row.keys()])
                    values = ", ".join(
                        f"'{value}'" if value is not None else "NULL" for value in row.values()
                    )
                    insert_stmt = f"""INSERT INTO {table_name} ({columns}) VALUES ({values});"""
                    # insert_stmt = f"INSERT INTO {table_name} (`{columns}`) VALUES ({values});"
                    records_dump.append(insert_stmt)
                    for pk in pks_values.keys():
                        pks_values[pk].append(row.get(pk))
                    
                    if len(records_dump)> size:
                        break
                
                
                if len(result.all()) < size:
                    # exp.select('*').from_(table_name).where(pred)
                    cond = text(' and '.join([f"`{pk}` not in {tuple(vals)}" for pk, vals in pks_values.items()]))
                    stmt = table.select().where(cond)
                    cursor = conn.execute(stmt)
                    result = cursor.fetchmany(size - len(result.all()))
                    for row in result:
                        row = row._asdict()
                        columns = ", ".join([f"`{k}`" for k in row.keys()])
                        # columns = ", ".join(row.keys())
                        values = ", ".join(
                            f"'{value}'" if value is not None else "NULL" for value in row.values()
                        )
                        # insert_stmt = f"""INSERT INTO {table_name} ("{columns}") VALUES ({values});"""
                        insert_stmt = f"""INSERT INTO {table_name} ({columns}) VALUES ({values});"""
                        records_dump.append(insert_stmt)
                        for pk in pks_values.keys():
                            pks_values[pk].append(row.get(pk))
                else:
                    records_dump = random.sample(records_dump, k= size)
                database_dump.extend(records_dump)
        
        
        self.create_database(schema_dump, database_dump, host_or_path= '', database= to)
        
        return schema_dump, database_dump
        
    def create_database(self, schemas: List[str], inserts: List[str], host_or_path, database, port = None, username = None, password = None, dialect = 'sqlite'):

        with self.get_connection(host_or_path, database, port, username, password, dialect) as conn:
            conn.create_tables(*schemas)
            for insert_stmt in inserts:
                conn.execute(insert_stmt, fetch= None, commit= True)