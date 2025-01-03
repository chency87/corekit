from typing import Union, List, Dict, Tuple, Set, Optional
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
        return str(exp.Literal.string(value))
    # "'" + value.replace("'", "''") + "'"  # Escape single quotes
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
               random_order = False, size = 10, quoted = True, dialect = 'sqlite'):
    
    ddls = DBManager().get_schema(original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect)
    schema, primary_keys, foreign_keys = unify_schema(ddls, dialect= dialect)
    sampled_stmts = sample_data_by_queries(schema= schema, queries= queries, random_order= random_order, size= size, dialect= dialect, quote= quoted)
    sample = get_sampled_data(sampled_stmts, original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect )
    ## ensure row size 
    stmts = ensure_row_size(schema, primary_keys, sample, size = size, quoted = quoted, dialect= dialect)
    if stmts:
        append_data = get_sampled_data(stmts,  original_host_or_path, original_database, original_port, original_username, original_password, dialect= dialect)
        for table_name, data in append_data.items():
            sample[table_name].extend(data)
    
    sample = ensure_data_dependency(schema, foreign_keys= foreign_keys, datasets= sample, size = size, host_or_path= original_host_or_path, database= original_database,\
                                    port = original_port, username= original_username, password= original_password, quoted= quoted, dialect= dialect)
    
    ## convert dataframe to SQL insert statements
    inserts = []
    for table_name in topo_tables(schema, foreign_keys):
        data = sample.get(table_name, [])
        columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted)) for col in schema.column_names(table_name)]
        table_identifier = exp.to_identifier(table_name, quoted= quoted)
        
        expressions = []

        for row in data:
            value = exp.tuple_(*row, dialect = dialect)
            expressions.append(value)
        if expressions:
            insert_stmt = exp.Insert(this = exp.Schema(this = exp.Table(this = table_identifier), expressions = columns), expression = exp.Values(expressions = expressions))
            inserts.append(insert_stmt.sql(dialect = dialect))

    if to_host_or_path is not None and to_database is not None:
        DBManager().create_database(schemas = schema, inserts= inserts,  host_or_path= to_host_or_path, database = to_database, port = to_port, username = to_username, password= to_password, dialect= dialect)

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
        table_names.add(from_.this.name.lower())
        joins = expression.args.get("joins")
        if joins:
            for join in joins:
                for tbl in join.find_all(exp.Table):
                    table_names.add(tbl.name.lower())
        for tbl_name in table_names:
        # for tbl in expression.find_all(exp.Table):
            stmt = expression.copy()
            columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted), table = tbl.alias) for col in schema.column_names(tbl_name, dialect= dialect)]
            stmt.set('expressions', columns)
            for keyword in ['distinct', 'order', 'offset', 'limit', 'group', 'having']:
                stmt.set(keyword, None)
            statements[tbl_name] =  stmt
    elif isinstance(expression, exp.Union):
        left = process_query(schema= schema, expression= expression.left, dialect= dialect, quoted= quoted, ctes= ctes)
        right = process_query(schema= schema, expression= expression.right, dialect= dialect, quoted= quoted, ctes= ctes)
        
        for table_name, stmt in left.items():
            tmp = stmt
            if table_name in right:
                tmp = exp.union(tmp, right.pop(table_name), dialect= dialect)
            statements[table_name] = tmp        
        for table_name, stmt in right.items():
            statements[table_name] = stmt
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
                values =  tuple([escape_value(value) for value in row.values()])
                # str(exp.tuple_(escape_value(value) for value in row.values()))
                #
                
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
            index = column_names.index(pk)
            pk_values[pk] = [row[index] for row in data]
        columns = [exp.Column(this = exp.to_identifier(col, quoted= quoted)) for col in column_names]
        stmt = exp.Select(expressions = columns).from_(exp.to_identifier(table_name, quoted= quoted)).limit(size - len(data))
        where = []
        for pk, values in pk_values.items():
            if not values:
                continue
            where.append(f'{pk} not in {str(tuple(values))}')
        if where:
            stmt = stmt.where(' and '.join(where))
        stmts[table_name] = stmt.sql(dialect = dialect)
    return stmts


def get_foreign_key_dependent_condition(foreign_keys: List, table_name, datasets: Dict[str, List], schema: MappingSchema) -> str:
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
    
    where = [f"{column} in {str(exp.tuple_(*condition))}" for column, condition in conditions.items() if condition]
    condition_str = " and ".join(where) if where else None
    
    return condition_str


def get_dependent_data(datasets: Dict[str, List], table_name, column_name, schema: MappingSchema):
    columns = schema.column_names(table_name)
    data = []
    for row in datasets.get(table_name, []):
        data.append(row[columns.index(column_name)])
    return data

def topo_tables(schema: MappingSchema, foreign_keys: List[ForeignKey]):
    ## calculate out degree of each table
    out_degree = {table_name : 0 for table_name in schema.mapping}
    for fk in foreign_keys:
        out_degree[fk.from_table] += 1

    sorted_table = dict(sorted(out_degree.items(), key=lambda item: item[1], reverse=True))

    return sorted_table

def ensure_data_dependency(schema: MappingSchema, foreign_keys: List[ForeignKey], datasets: Dict[str, List], size, \
                           host_or_path: str, database: str, port = None, username = None, password = None,
                           quoted = True, dialect = 'sqlite'):
    
    sorted_table = topo_tables(schema, foreign_keys)

    visit = set(datasets.keys())
    for table_name in sorted_table:
        if table_name in visit:
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
        
