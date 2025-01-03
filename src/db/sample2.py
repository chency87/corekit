from typing import Union, List, Dict, Tuple, Set, Optional, overload
from collections import defaultdict, OrderedDict, deque
from sqlglot import exp, parse_one, parse
from sqlglot.schema import ensure_schema, MappingSchema
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
def escape_value(value):
    """Escape single quotes and special characters in SQL strings."""
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        return value.replace(':', '\:')   # Escape single quotes
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
def sample_small_database(queries: List[str], original_host_or_path, original_database, original_port= None, original_username= None, original_password= None,\
               to_host_or_path = None, to_database = None, to_port = None, to_username = None, to_password = None, \
               random_order = False, size = 10, quoted = True, dialect = 'sqlite', remove_table = False, max_row_size = False):
    
    ddls = DBManager().get_schema(original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect)
    schema, primary_keys, foreign_keys = unify_schema(ddls, dialect= dialect)
    if remove_table:
        schema, primary_keys = remove_tables(schema, primary_keys= primary_keys, queries= queries, dialect= dialect)

    sampled_stmts = sample_data_by_queries(schema= schema, queries= queries, random_order= random_order, size= size, dialect= dialect, quote= quoted)

    # print(sampled_stmts)

    sample = get_sampled_data(sampled_stmts, original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect )
    ## ensure row size 
    if max_row_size:
        stmts = ensure_row_size(schema, primary_keys, sample, size = size, quoted = quoted, dialect= dialect)
        if stmts:
            append_data = get_sampled_data(stmts, original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect)
            for table_name, data in append_data.items():
                sample[table_name].extend(data)
    
    sample = ensure_data_dependency(schema, foreign_keys= foreign_keys, datasets= sample, size = size, host_or_path= original_host_or_path, database= original_database,\
                                    port = original_port, username= original_username, password= original_password, quoted= quoted, dialect= dialect)
    
    ## convert dataframe to SQL insert statements
    inserts = unify_insert_stmt(schema= schema, foreign_keys= foreign_keys, datasets= sample, quoted= quoted, dialect= dialect)

    if to_host_or_path is not None and to_database is not None:
        create_table_stmts = []
        for table_name, column_defs in schema.mapping.items():
            columns = [exp.ColumnDef(this = exp.to_identifier(column_name, quoted = True), kind = exp.DataType.build(column_typ)) for column_name, column_typ in column_defs.items()]
            ddl = exp.Create(this = exp.Schema(this = exp.to_identifier(table_name, quoted = True) , expressions = columns), exists = True, kind = 'TABLE')
            create_table_stmts.append(ddl.sql(dialect= dialect))
        with DBManager().get_connection(host_or_path= to_host_or_path, database = to_database, port = to_port, username = to_username, password= to_password, dialect= dialect) as conn:
            conn.create_tables(*create_table_stmts)
            for insert_stmt, data in inserts:
                conn.insert(insert_stmt, data)

def unify_insert_stmt(schema: MappingSchema, foreign_keys: List[ForeignKey], datasets, quoted, dialect):
    inserts = []
    for table_name in topo_tables(schema, foreign_keys):
        data = datasets.get(table_name, [])
        columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted)) for col in schema.column_names(table_name)]
        table_identifier = exp.to_identifier(table_name, quoted= quoted)
        # placeholders = [exp.Placeholder(this = str(col)) for col in schema.column_names(table_name)]

        if data:
            values = []
            for row in data:
                values.append(exp.tuple_(*(exp.convert(escape_value(row[col])) for col in schema.column_names(table_name))))

            insert_stmt = exp.Insert(this = exp.Schema(this = exp.Table(this = table_identifier), expressions = columns), expression = exp.Values(expressions = values))
            
            inserts.append((insert_stmt.sql(dialect = dialect), None))
    return inserts


def unify_schema(ddls: str, dialect) -> Tuple[MappingSchema, Dict[str, Set], List[ForeignKey]]:

    '''Convert SQL create statements to Dict. Return:
        MappingSchema({'tbl': {'col': 'typ'}}), PrimaryKeys, ForeignKeys
    '''
    schema = {}
    foreign_keys: List[ForeignKey] = []
    primary_keys: Dict[str, Set] = defaultdict(set)
    for expr in parse(ddls, dialect= dialect):
        columns = OrderedDict()
        obj = expr.this
        tbl_name = obj.this.name.lower()
        for column_def in obj.expressions:
            if isinstance(column_def, exp.ColumnDef):
                columns[column_def.alias_or_name] = str(column_def.kind)
                for constraint in column_def.constraints[:]:
                    if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                        primary_keys[tbl_name].add(column_def.alias_or_name.lower())
            elif isinstance(column_def, exp.PrimaryKey):
                primary_keys[tbl_name].update([item.alias_or_name.lower() for item in column_def.expressions])
            elif isinstance(column_def, exp.ForeignKey):
                from_tbl = column_def.args.get('reference').find(exp.Schema)
                from_tbl_name = from_tbl.this.name.lower()
                from_column = from_tbl.expressions[0].name.lower()
                to_tbl_name = tbl_name
                to_column = column_def.expressions[0].this.lower()
                foreign_keys.append(ForeignKey(from_table= from_tbl_name, from_column = from_column, to_table= to_tbl_name, to_column= to_column))
        schema[tbl_name] = columns
    schema = ensure_schema(schema, dialect = dialect)
    return schema, primary_keys, foreign_keys



def remove_tables(schema: MappingSchema, primary_keys, queries: List[str], dialect = 'sqlite'):
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
    new_primary_keys = {}
    
    for tbl in schema.mapping:
        if tbl.lower() in tables:
            new_schema[tbl.lower()] = schema.mapping[tbl.lower()]

        if tbl.lower() in primary_keys:
            new_primary_keys[tbl.lower()] = primary_keys[tbl.lower()]

    return ensure_schema(new_schema, dialect = dialect), new_primary_keys

def sample_data_by_queries(schema: MappingSchema, queries: Union[str, List[str]], random_order = True, size = 10, dialect = 'sqlite', quote = True):
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
        try:
            expr = qualify.qualify(parse_one(str(query), dialect = dialect), schema= schema, quote_identifiers= quote, qualify_columns= False)
        except Exception as e:
            expr = parse_one(str(query), dialect = dialect)
        statements = process_query(schema= schema, expression= expr, dialect= dialect, quoted= quote)
        for table_name, stmt in statements.items():
            if table_name in samples:
                samples[table_name] = exp.union(samples[table_name], stmt, dialect= dialect)
            else:
                samples[table_name] = stmt
    for table_name in samples:
        samples[table_name] = samples[table_name].limit(size)
        if random_order:
            samples[table_name] = samples[table_name].order_by(exp.func('random', dialect= dialect))
        samples[table_name] = samples[table_name].sql(dialect = dialect)
    return samples


def union_statements(left: Dict, right: Dict, dialect):

    statements = {}
    for table_name, stmt in left.items():
        tmp = stmt
        if table_name in right:
            tmp = exp.union(tmp, right.pop(table_name), dialect= dialect)
        statements[table_name] = tmp        
    for table_name, stmt in right.items():
        statements[table_name] = stmt

    return statements



def process_query(schema: MappingSchema, expression: exp.Expression, dialect: str, quoted: bool,\
                  ctes: Optional[Dict[str, exp.Expression]] = None) -> Dict[str, exp.Select]:
    statements = {}
    ctes = ctes or {}
    expression = expression.unnest()
    with_ = expression.args.get("with")
    if with_:
        ctes = ctes.copy()
        for cte in with_.expressions:
            ctes[cte.alias] = cte.this
    
    if isinstance(expression, exp.Select):
        table_names = set()
        from_ = expression.args.get('from')
        if not from_:
            for expr in expression.args.get('expressions'):
                for subquery in expr.find_all(exp.Select):
                    tmp = process_query(schema= schema, expression= subquery, dialect= dialect, quoted= quoted, ctes= ctes)
                    statements = union_statements(statements, tmp, dialect= dialect)

        elif isinstance(from_.this, exp.Subquery):
            tmp = process_query(schema= schema, expression= from_.this, dialect= dialect, quoted= quoted, ctes= ctes)
            statements = union_statements(statements, tmp, dialect= dialect)
        else:
            table_names.add(from_.this)
            joins = expression.args.get("joins")
            if joins:
                for join in joins:
                    if isinstance(join.this, exp.Subquery):
                        tmp = process_query(schema= schema, expression= join.this, dialect= dialect, quoted= quoted, ctes= ctes)
                        statements = union_statements(statements, tmp, dialect= dialect)
                    else:
                        for tbl in join.find_all(exp.Table):
                            table_names.add(tbl)
            for tbl in table_names:
                tbl_name = tbl.name.lower()
                stmt = expression.copy()
                columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted), table = exp.to_identifier(tbl.alias_or_name, quoted= quoted)) for col in schema.column_names(tbl_name)]
                stmt.set('expressions', columns)
                for keyword in ['distinct', 'order', 'offset', 'limit', 'group', 'having']:
                    stmt.set(keyword, None)
                if tbl_name in statements:
                    statements[tbl_name] =  exp.union(statements[tbl_name], stmt)
                else:
                    statements[tbl_name] =  stmt
                    
    elif isinstance(expression, exp.Union):
        left = process_query(schema= schema, expression= expression.left, dialect= dialect, quoted= quoted, ctes= ctes)
        right = process_query(schema= schema, expression= expression.right, dialect= dialect, quoted= quoted, ctes= ctes)        
        statements = union_statements(statements,  union_statements(left, right=right, dialect= dialect), dialect= dialect)
    elif isinstance(expression, exp.Subquery):
        tmp = process_query(schema= schema, expression= expression.this, dialect= dialect, quoted= quoted, ctes= ctes)
        statements = union_statements(statements, tmp, dialect= dialect)
    else:
        raise ValueError(f'Unsupported expression: {expression}')
    
    return statements

def get_sampled_data(stmts: Dict[str, str], host_or_path: str, database: str, port = None, username = None, password = None, dialect = 'sqlite') -> Dict[str, List]:
    datasets = defaultdict(list)
    with DBManager().get_connection(host_or_path, database, port, username, password, dialect) as conn:
        for table_name, stmt in stmts.items():
            results = conn.execute(stmt, fetch= 'all')
            for row in results:
                row = row._asdict()
                values = {name.lower(): value for name, value in row.items()}
                datasets[table_name].append(values)
    return datasets


def ensure_row_size(schema: MappingSchema, primary_keys, datasets, size, quoted = True, dialect = 'sqlite'):
    '''
        ensure that the count of rows in each table satisfies the size requirement.
        Specifically, we coudl build a query to select from table. the query should contain a condition that pk not in existing values
    '''
    stmts = {}
    for table_name in datasets:
        data = datasets[table_name]
        if len(data) >= size:
            continue
        pk_values = {}
        column_names = schema.column_names(table_name)
        for pk in primary_keys[table_name]:
            pk_values[pk] = [row[pk] for row in data]
        columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted)) for col in column_names]
        stmt = exp.Select(expressions = columns).from_(exp.to_identifier(table_name, quoted= quoted)).limit(size - len(data))
        where = []
        for pk, values in pk_values.items():
            if not values:
                continue
            
            where.append(exp.Not(this = exp.In(this = exp.to_column(pk, quoted= quoted), expressions = [exp.convert(c) for c in values])))
        
        if where:
            where = reduce(lambda x, y : exp.And(this = x, expression =y), where)
            stmt = stmt.where(where)
        
        stmts[table_name] = stmt.sql(dialect = dialect)
    return stmts


def get_foreign_key_dependent_condition(foreign_keys: List, table_name, datasets: Dict[str, List], schema: MappingSchema) -> exp.Condition:
    conditions = {}
    for fk in foreign_keys:
        if fk.from_table == table_name:
            if fk.from_column not in conditions:
                conditions[fk.from_column] = set()
            
            data = get_dependent_data(datasets= datasets, table_name= fk.from_table, column_name= fk.from_column, schema= schema)
            conditions[fk.from_column].update(data)
            to_data = get_dependent_data(datasets= datasets, table_name= fk.to_table, column_name= fk.to_column, schema= schema)
            conditions[fk.from_column].update(to_data)

        if fk.to_table == table_name:
            if fk.to_column not in conditions:
                conditions[fk.to_column] = set()
            data = get_dependent_data(datasets= datasets, table_name= fk.from_table, column_name= fk.from_column, schema= schema)
            conditions[fk.to_column].update(data)
            to_data = get_dependent_data(datasets= datasets, table_name= fk.to_table, column_name= fk.to_column, schema= schema)
            conditions[fk.to_column].update(to_data)
    
    # where = [f"{column} in {exp.tuple_(*(exp.convert(c) for c in condition)).sql()}" for column, condition in conditions.items() if condition]

    # for column, dep_data in conditions.items():
    #     if dep_data:
    #         exp.In(this = exp.to_column(column, quoted= True), expressions = [exp.convert(c) for c in dep_data])

    where = [exp.In(this = exp.to_column(column, quoted= True), expressions = [exp.convert(c) for c in dep_data]) for column, dep_data in conditions.items() if dep_data]
    condition = None
    if where:
        condition = reduce(lambda x, y : exp.And(this = x, expression =y), where)

    # condition_str = " and ".join(where) if where else None
    
    return condition


def get_dependent_data(datasets: Dict[str, List], table_name, column_name, schema: MappingSchema):
    data = []
    for row in datasets.get(table_name, []):
        data.append(row[column_name])
    return data

def topo_tables(schema: MappingSchema, foreign_keys: List[ForeignKey]):
    ## calculate out degree of each table
    out_degree = {table_name : 0 for table_name in schema.mapping}
    for fk in foreign_keys:
        if fk.from_table in out_degree:
            out_degree[fk.from_table] += 1
    sorted_table = dict(sorted(out_degree.items(), key=lambda item: item[1], reverse=True))
    return sorted_table

def ensure_data_dependency(schema: MappingSchema, foreign_keys: List[ForeignKey], datasets: Dict[str, List], size, \
                           host_or_path: str, database: str, port = None, username = None, password = None,
                           quoted = True, dialect = 'sqlite'):
    sorted_table = topo_tables(schema, foreign_keys)

    visit = set(datasets.keys())
    for table_name in sorted_table:
        if table_name in visit or table_name.lower() not in schema.mapping:
            continue
        
        where = get_foreign_key_dependent_condition(foreign_keys, table_name, datasets, schema)
        columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted)) for col in schema.column_names(table_name)]
        stmt = exp.Select(expressions = columns).from_(exp.to_identifier(table_name, quoted= quoted)).limit(size)

        if where:
            stmt = stmt.where(where)
        stmt = stmt.sql(dialect= dialect)
        
        data = get_sampled_data({table_name: stmt}, host_or_path, database= database, port= port, username= username, password= password, dialect= dialect)
        for k, v in data.items():
            datasets[k] = v
        visit.add(table_name)
    return datasets
        
