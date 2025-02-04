from typing import Union, List, Dict, Tuple
from collections import defaultdict, OrderedDict, deque
from sqlglot import exp, parse_one, parse
from sqlglot.optimizer import qualify
from datetime import date, datetime
from dataclasses import dataclass
from functools import reduce
import logging

from .db_manage import DBManager

logger = logging.getLogger('src.db.sample')


@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    def __hash__(self):
        return hash(self.from_table)
def escape_value(value):
    """Escape single quotes and special characters in SQL strings."""
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"  # Escape single quotes
    elif isinstance(value, int):
        return int(value)
    elif isinstance(value, float):
        return float(value)
    elif isinstance(value, date):
        return escape_value(value.strftime('%Y-%m%d'))
    elif isinstance(value, datetime):
        return escape_value(value.strftime('%Y-%m%d %H:%M%S'))
    else:
        return value  # Use repr for other types
    
def sample_small_db(queries: List[str], original_host_or_path, original_database, original_port= None, original_username= None, original_password= None,\
               to_host_or_path = None, to_database = None, to_port = None, to_username = None, to_password = None, \
               random_order = True, size = 10, quote = True, dialect = 'sqlite', remove_table = False) -> Tuple[Dict, List[str]]:
    '''
        Sample a small instance from original database to satisfy queries.
    '''
    ddls = DBManager().get_schema(original_host_or_path, original_database, original_port, original_username, original_password, dialect)

    schema, foreign_keys = jsonify_ddl(ddls, dialect= dialect)
    if remove_table:
        schema = remove_tables(schema, queries)

    schema, stmts = sample_data_from_original_db(schema, queries, random_order = random_order, dialect= dialect, size= size, quote= quote)

    inserts = []
    processed = set()
    dependent_data = defaultdict(dict)
    with DBManager().get_connection(original_host_or_path, original_database, original_port, original_username, original_password, dialect) as conn:
        for table_name, stmt in stmts.items():
            results = conn.execute(stmt, fetch= 'all')
            for row in results:
                row = row._asdict()
                columns = ", ".join([f"`{k}`" for k in row.keys()])
                values = ', '.join([str(escape_value(value)) for value in row.values()])
                insert_stmt = f"INSERT INTO `{table_name}` ({columns}) VALUES ({values});"
                inserts.append(insert_stmt)
                update_foreign_key_dependency(foreign_keys, table_name, row, dependent_data)
            processed.add(table_name)

    for table_name, column_defs in schema.items():
        if table_name not in processed:
            columns = [exp.Column(this = exp.to_identifier(col, quoted= quote)) for col in column_defs]
            select = exp.select(*columns, dialect= dialect).from_(exp.to_identifier(table_name, quoted= quote)).limit(size)
            where = get_foreign_key_dependent_condition(foreign_keys, table_name, dependent_data)
            if where:
                select = select.where(where)
            stmt = select.sql(dialect = dialect)
            with DBManager().get_connection(original_host_or_path, original_database, original_port, original_username, original_password, dialect) as conn:
                results = conn.execute(stmt, fetch= 'all')
                for row in results:
                    row = row._asdict()
                    columns = ", ".join([f"`{k}`" for k in row.keys()])
                    values = ', '.join([str(escape_value(value)) for value in row.values()])
                    insert_stmt = f"INSERT INTO `{table_name}` ({columns}) VALUES ({values});"
                    inserts.append(insert_stmt)
                    update_foreign_key_dependency(foreign_keys, table_name, row, dependent_data)
            processed.add(table_name)
            
    if to_host_or_path is not None and to_database is not None:
        DBManager().create_database(schemas = ddls, inserts= inserts,  host_or_path= to_host_or_path, database = to_database, port = to_port, username = to_username, password= to_password, dialect= dialect)
    return schema, stmts

def get_foreign_key_dependent_condition(foreign_keys: List, table_name, dependent_data) -> str:
    conditions = {}
    for fk in foreign_keys:
        if fk.from_table == table_name:
            if fk.from_column not in conditions:
                conditions[fk.from_column] = set()
            if fk.from_column in dependent_data[fk.from_table]:
                conditions[fk.from_column].update(dependent_data[fk.from_table][fk.from_column])
            if fk.to_table in dependent_data and fk.to_column in dependent_data[fk.to_table]:
                conditions[fk.from_column].update(dependent_data[fk.to_table][fk.to_column])

        if fk.to_table == table_name:
            if fk.to_column not in conditions:
                conditions[fk.to_column] = set()
            if fk.to_column in dependent_data[fk.to_column]:
                conditions[fk.to_column].update(dependent_data[fk.to_table][fk.to_column])
            if fk.from_column in dependent_data[fk.from_table]:
                conditions[fk.to_column].update(dependent_data[fk.from_table][fk.from_column])
    where = [f"{column} in {str(tuple(condition))}" for column, condition in conditions.items() if condition]
    condition_str = " and ".join(where) if where else None
    
    return condition_str

def update_foreign_key_dependency(foreign_keys, table_name, row: Dict, dependent_data):
    '''
        Update table dependence based on query execution results. if a table connects with other tables, we should cache its execution results.
    '''
    for fk in foreign_keys:
        if fk.from_table == table_name:
            if fk.from_column not in dependent_data[fk.from_table]:
                dependent_data[fk.from_table][fk.from_column]= []
        if fk.to_table == table_name:
            if fk.to_column not in dependent_data[fk.to_column]:
                dependent_data[fk.to_table][fk.to_column] = []

    if table_name in dependent_data:
        for col in dependent_data[table_name]:
            dependent_data[table_name][col].append(escape_value(row.get(col)))


def sample_data_from_original_db(schema: Dict[str, Dict[str, str]], queries: Union[str, List[str]], random_order = True, size = 10, dialect = 'sqlite', quote = True) -> Tuple[Dict, Dict[str, str]]:
    '''
        Sample a small instance to satisfy queries. return a tuple <schema, Dict of sample queries>
        schema: {tbl: {col: typ}}
        sample queries:
        {tbl: sample_query}
    '''
    if not isinstance(queries, list):
        queries = [queries]

    samples = {}
    for query in queries:
        statements = extract_predicates3(schema, query= query, size= size, random_order= random_order, quote= quote)
        for tbl, stmt in statements.items():
            if tbl in samples:
                samples[tbl] = exp.union(samples[tbl], stmt, dialect= dialect)
            else:
                samples[tbl] = stmt
    for tbl in samples:
        samples[tbl] = samples[tbl].limit(size)
        samples[tbl] = samples[tbl].sql(dialect = dialect)
    return schema, samples


def jsonify_ddl(ddls, dialect = 'sqlite'):
    '''Convert SQL create statements to Dict. Return:
        {'tbl': {'col': 'typ'} }, ForeignKeys
    '''
    schema = {}
    foreign_keys: List[ForeignKey] = []
    for expr in parse(ddls, dialect= dialect):
        columns = OrderedDict()        
        obj = expr.this
        tbl_name = obj.this.name.lower()
        for column_def in obj.expressions:
            if isinstance(column_def, exp.ColumnDef):
                columns[column_def.alias_or_name] = str(column_def.kind)
            elif isinstance(column_def, exp.ForeignKey):
                from_tbl = column_def.args.get('reference').find(exp.Schema)
                from_tbl_name = from_tbl.this.name.lower()
                from_column = from_tbl.expressions[0].name
                to_tbl_name = tbl_name
                to_column = column_def.expressions[0].this
                foreign_keys.append(ForeignKey(from_table= from_tbl_name, from_column = from_column, to_table= to_tbl_name, to_column= to_column))
        schema[tbl_name] = columns
    return schema, foreign_keys


def remove_tables(schema: Dict[str, Dict[str, str]], queries: List[str], dialect = 'sqlite'):
    '''
        Remove irrelevant tables from schema. Return:
        {'tbl': {'col': 'typ'} }
    '''
    tables = set()
    for query in queries:
        expr = parse_one(sql = str(query), dialect = dialect)
        for tbl in expr.find_all(exp.Table):
            tables.add(tbl.this.name.lower())

    new_schema = {}
    for tbl, column_defs in schema.items():
        if tbl.lower() in tables:
            new_schema[tbl.lower()] = {column_name: column_typ for column_name, column_typ in column_defs.items()}
    return new_schema

def remove_columns(schema: Dict[str, Dict[str, str]], queries: List[str], dialect = 'sqlite'):
    '''
        Remove irrelevant columns from schema. Return:
        {'tbl': {'col': 'typ'} }
    '''
    columns = set()
    for query in queries:
        expr = parse_one(sql = str(query), dialect = dialect)
        for column in expr.find_all(exp.Column):
            columns.add(column.name)
    new_schema = {}
    for tbl, column_defs in schema.items():
        new_schema[tbl] = {column_name: column_typ for column_name, column_typ in column_defs.items() if column_name in columns}
    return new_schema


def remove_table_columns(schema: Dict[str, Dict[str, str]], queries: List[str], dialect = 'sqlite'):
    '''
        Remove all tables and columns not exits in queries from schema. Return:
        {'tbl': {'col': 'typ'} }
    '''
    tables = set()
    columns = set()
    for query in queries:
        expr = parse_one(sql = str(query), dialect = dialect)
        for tbl in expr.find_all(exp.Table):
            tables.add(tbl.this.name)
        for column in expr.find_all(exp.Column):
            columns.add(column.name)    
    new_schema = {}
    for tbl, column_defs in schema.items():
        if tbl in tables:
            new_schema[tbl] = {column_name: column_typ for column_name, column_typ in column_defs.items() if column_name in columns}

    return new_schema
    

def merge_samples_by_table(table_alias, statements: Dict, dialect = 'sqlite'):
    stmts = {}
    for tbl_name, tbl_alias in table_alias.items():
        stmt = statements[tbl_alias.pop()]
        for alias in tbl_alias:
            stmt = exp.union(stmt, statements[alias])
        stmts[tbl_name] = stmt
    return stmts

def extract_predicates3(schema: Dict[str, Dict[str, str]], query: str, dialect = 'sqlite', random_order = True, size = 10, quote = True):
    '''
        Extract predicates from SQL qeury. E.g.
        >>> SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18

        >>> SELECT T1.* FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18
        >>> SELECT T2.* FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0
    '''
    # expr = qualify.qualify(parse_one(str(query), dialect = dialect), schema= schema, quote_identifiers= quote, qualify_columns= False)
    expr = parse_one(str(query), dialect = dialect)
    selects = expr.find_all(exp.Select) ## main select and subqueries
    statements = {}
    table_alias = defaultdict(list)
    for select in selects:
        for tbl in select.find_all(exp.Table):
            tbl_name = tbl.name.lower()
            table_alias[tbl_name].append(tbl.alias_or_name)
            col = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = tbl.alias) for col in schema[tbl_name]]
            stmt = select.copy()
            stmt.set('expressions', col)
            stmt.set('order', None)
            stmt.set('limit', None)
            stmt.set('group', None)
            stmt.set('having', None)
            statements[str(tbl.alias_or_name)] =  stmt
    
    return merge_samples_by_table(table_alias, statements)



def extract_predicates2(schema: Dict[str, Dict[str, str]], query: str, dialect = 'sqlite', random_order = True, size = 10, quote = True):
    '''
        Extract predicates from SQL qeury. E.g.
        >>> SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18

        >>> SELECT T1.* FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18
        >>> SELECT T2.* FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0

    '''
    
    expr = qualify.qualify(parse_one(str(query), dialect = dialect), schema= schema, quote_identifiers= quote)
    
    selects = expr.find_all(exp.Select) ## main select and subqueries

    statements = {}
    table_alias = defaultdict(list)

    for select in selects:
        
        table_conditions = defaultdict(list)

        for tbl in select.find_all(exp.Table):
            table_alias[tbl.name].append(str(tbl.alias))
            col = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = tbl.alias) for col in schema[tbl.name]]
            statements[str(tbl.alias)] = exp.Select(expressions = col).limit(size)
        if random_order:
            for alias in statements:
                statements[alias] = statements[alias].order_by(exp.func('random', dialect= dialect))
        from_ = select.args.get('from')
        for alias in statements:
            statements[alias] = statements[alias].from_(from_)

        joins = select.args.get("joins")
        if joins:
            for join in joins:
                for alias in statements:
                    statements[alias] = statements[alias].join(join)
        
        where = select.args.get("where")
        if where:
            for pred in where.find_all(exp.Predicate):
                if 0 < len(exp.column_table_names(pred)) < 2:
                    tbl = exp.column_table_names(pred).pop()
                    table_conditions[tbl].append(pred)

            for alias in statements:
                conditions = table_conditions.get(alias, [])
                if conditions:
                    condition = reduce(lambda x, y : exp.And(this = x, expression =y), conditions)
                    statements[alias] = statements[alias].where(condition)

    return merge_samples_by_table(table_alias, statements)





def extract_predicates(schema: Dict[str, Dict[str, str]], queries: List[str], dialect = 'sqlite', size  = 5, quote = True):
    '''
        Extract predicates from queries and generate a SELECT stmt.
    '''
    # a tabel might have multiple alias, we need to random data from all alias
    table_alias = defaultdict(list)
    table_conditions = defaultdict(list)
    table_joins = defaultdict(dict)

    for query in queries:
        expr = qualify.qualify(parse_one(str(query), dialect = dialect), schema= schema, quote_identifiers= quote)
        for tbl in expr.find_all(exp.Table):
            table_alias[tbl.name].append(str(tbl.alias))

        selects = expr.find_all(exp.Select)

        for select in selects:
            from_ = select.args.get('from')
            joins = select.args.get("joins")
            if joins:
                for join in joins:
                    for join_tbl in join.find_all(exp.Column):
                        table_joins[str(join_tbl.table)]['from'] = from_
                        if 'join' not in table_joins[str(join_tbl.table)]:
                            table_joins[str(join_tbl.table)]['join'] = set()
                        table_joins[str(join_tbl.table)]['join'].update(joins)
       

        for select in selects:
            where = select.args.get("where")
            if where:
                for pred in where.find_all(exp.Predicate):
                    if 0 < len(exp.column_table_names(pred)) < 2:
                        tbl = exp.column_table_names(pred).pop()
                        table_conditions[tbl].append(pred)
    # logger.info(table_joins)
    return table_alias, table_conditions, table_joins


def build_with_clause(stmt: exp.Expression, alias: exp.TableAlias):
    return exp.CTE(this = stmt, alias = alias)
def build_main_part(parts):
    return reduce(lambda x, y : exp.union(x, y, distinct= False), parts)

def build_additional_parts(columns: List[str], table_name: str, tsize, avail_steps: List[exp.CTE], primary_keys: List[str],  quote = True):
    '''
        Build a cluase of CTE to select more rows from table to satisfy size constraint. Return:
            # step2 AS (
            #     -- Step 2: Select additional random rows excluding those in previous steps
            #     SELECT *
            #     FROM frpm
            #     WHERE CDSCode NOT IN (SELECT CDSCode FROM step1)
            #     ORDER BY RANDOM()
            #     LIMIT 20 - (SELECT COUNT(*) FROM step1)
            # )
    '''
    step_alias =  exp.TableAlias(this = exp.to_identifier( f'step{len(avail_steps)}'))
    addsitional_rows_alias = exp.TableAlias(this = exp.to_identifier( f'additional_{len(avail_steps)}'))
    table = exp.to_table(table_name, alias = addsitional_rows_alias)
    
    column_defs = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = addsitional_rows_alias.this) for col in columns]

    ## total count of previous ctes
    counts = [exp.Literal.number(tsize)]
    for cte in avail_steps:
        counts.append(exp.select('count(*)').from_(cte.alias_or_name))
    limit_ = reduce(lambda x, y : exp.Sub(this = x, expression = exp.Subquery(this = y) if not isinstance(y, exp.Subquery) else y), counts)

    
    body_columns = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = step_alias.this) for col in columns]
    e = exp.Select(expressions = body_columns).from_(step_alias)
    return build_with_clause(stmt= exp.Select(expressions = column_defs).from_(table).order_by(exp.func('RANDOM')).limit(limit_), alias= step_alias), e


def build_query(schema, table_alias: Dict[str, List[str]], table_conditions: Dict[str, exp.Predicate], table_joins:Dict[str, exp.Join], dialect = 'sqlite', size = 5, quote = True):

    steps = defaultdict(list)
    bodies =  defaultdict(list)

    for table_name, list_alias in table_alias.items():
        for index, alias in enumerate(list_alias):
            alias_identifier =  exp.to_identifier(alias)
            table = exp.to_table(table_name, alias = exp.TableAlias(this = alias_identifier))

            col = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = alias_identifier) for col in schema[table_name]]
            stmt = exp.Select(expressions = col)

            if table_joins.get(alias, {}):
                from_ = table_joins.get(alias).get('from')
                stmt = stmt.from_(from_)
                
                for join in table_joins.get(alias).get('join'):
                    stmt = stmt.join(join)
                    # stmt = stmt.from_(join)
            else:
                stmt = stmt.from_(table)

            conditions = table_conditions.get(alias, [])
            if conditions:
                where = reduce(lambda x, y : exp.And(this = x, expression =y), conditions)
                stmt = stmt.where(where)
            
                
            step_alias = exp.TableAlias(this = exp.to_identifier( f'step{index}'))

            steps[table_name].append(build_with_clause(stmt, alias= step_alias))
            ## process body queries
            body_columns = [exp.Column(this = exp.to_identifier(col, quoted= quote), table = step_alias.this) for col in schema[table_name]]
            bodies[table_name].append(exp.Select(expressions = body_columns).from_(step_alias))
    queries = {}
    for table_name, parts in bodies.items():
        # additional_cte, additional_main_part = build_additional_parts(list(schema[table_name].keys()), table_name= table_name, tsize= size, avail_steps= steps[table_name], primary_keys= [])
        # parts_ = [*parts, additional_main_part]
        # ctes = [*steps[table_name], additional_cte]
        ctes = steps[table_name]
        parts_ = parts
        body = build_main_part(parts_).limit(size)
        body.set('with', exp.With(expressions = ctes))
        queries[table_name] = body.sql(dialect= dialect)
    return queries




def jsonify_ddl2(ddls, dialect = 'sqlite'):
    '''Convert SQL create statements to Dict. Return:
        {'tbl': {'col': 'typ'} }
    '''

    dependency = defaultdict(list)

    fks = []
    
    out_degree = defaultdict(int)

    schema = {}
    for expr in parse(ddls, dialect= dialect):
        columns = OrderedDict()
        
        obj = expr.this
        tbl_name = obj.this.name.lower()
        out_degree[tbl_name] = out_degree.get(tbl_name, 0)
        for column_def in obj.expressions:
            if isinstance(column_def, exp.ColumnDef):
                columns[column_def.alias_or_name] = str(column_def.kind)
            elif isinstance(column_def, exp.ForeignKey):
                from_tbl = column_def.args.get('reference').find(exp.Schema)
                from_tbl_name = from_tbl.this.name.lower()
                from_column = from_tbl.expressions[0].name
                to_tbl_name = tbl_name
                to_column = column_def.expressions[0].this
                fks.append(ForeignKey(from_table= from_tbl_name, from_column = from_column, to_table= to_tbl_name, to_column= to_column))
                out_degree[from_tbl_name] += 1
        schema[tbl_name.lower()] = columns
    
    # print(out_degree)
    # queue = deque([table for table in out_degree if out_degree[table] == 0])

    # sorted_tables = []

    # while queue:
    #     table = queue.popleft()
    #     sorted_tables.append(table)

    #     for fk in fks:
    #         if out_degree[fk.from_table] == 0:
    #             queue.append(fk.from_table)
    #         if fk.from_table == table:
    #             out_degree[fk.from_table] -= 1

    return schema

# def topo_tables(ddls: str):
#     '''
#     topo sort schema and extract dependency
#     '''
#     visited = {k: False for k in self._tables.keys()}
#     q = deque([table_name for table_name in list(self._tables.keys() ) if table_name not in list(self.foreign_keys.keys()) or not self.foreign_keys.get(table_name, [])])
#     topo_order = []
#     while q:
#         table_name = q.popleft()
#         topo_order.append(table_name)
#         visited[table_name] = True
#         for tbl_name, fks in self.foreign_keys.items():
#             if visited[tbl_name]:
#                 continue
#             if all([fk.args.get('reference').find(exp.Table).name == tbl_name or visited[fk.args.get('reference').find(exp.Table).name] for fk in fks ]):
#                 if tbl_name not in q: q.append(tbl_name)        
#     return [self._tables[tbl_name] for tbl_name in topo_order]
