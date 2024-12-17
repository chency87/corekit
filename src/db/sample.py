from typing import Union, List, Dict, Tuple
from collections import defaultdict
from sqlglot import exp, parse_one, parse
from sqlglot.optimizer import qualify
from functools import reduce
import logging

logger = logging.getLogger('src.db.sample')


def sample_from_original_db(ddls: str, queries: Union[str, List[str]], size = 10, dialect = 'sqlite', quote = True) -> Tuple[Dict, Dict[str, str]]:

    '''
        Sample a small instance to satisfy queries. return a tuple <schema, Dict of sample queries>
        schema: {tbl: {col: typ}}
        sample queries:
        {tbl: sample_query}
    '''

    logger.warning('we should consider databasee constraints in the future')

    if not isinstance(queries, list):
        queries = [queries]

    schema = jsonify_ddl(ddls, dialect= dialect)

    schema = remove_table_columns(schema, queries = queries )
    # logger.info(schema)

    table_alias, table_condition, table_joins = extract_predicates(schema, queries, dialect= dialect, size= size, quote= quote)

    steps = build_query(schema= schema, table_alias= table_alias, table_conditions= table_condition, table_joins = table_joins, dialect= dialect, size = size, quote = quote)
    return schema, steps


def jsonify_ddl(ddls, dialect = 'sqlite'):
    '''
    Convert SQL create statements to Dict. Return:
        {'tbl': {'col': 'typ'} }
    '''
    schema = {}
    for expr in parse(ddls, dialect= dialect):
        tbl_name = expr.find(exp.Table).alias_or_name
        columns = {}
        for column_def in expr.find_all(exp.ColumnDef):
            columns[column_def.alias_or_name] = str(column_def.kind)
        schema[tbl_name] = columns
    return schema

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
    

def extract_predicates(schema, queries: List[str], dialect = 'sqlite', size  = 5, quote = True):
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
        ## process Join Parts
        for join in expr.find_all(exp.Join):
            # logger.info(join.find_ancestor(exp.Select).args.get('from'))
            from_ = join.find_ancestor(exp.Select).args.get('from')

            for ident in join.find_all(exp.Column):
                table_joins[str(ident.table)]['from'] = from_
                if 'join' not in table_joins[str(ident.table)]:
                    table_joins[str(ident.table)]['join'] = []
                table_joins[str(ident.table)]['join'].append(join)

        selects = expr.find_all(exp.Select)

        for select in selects:
            where = select.args.get("where")
            
            if where:
                # print(repr(where))
                ## build where parts
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
        additional_cte, additional_main_part = build_additional_parts(list(schema[table_name].keys()), table_name= table_name, tsize= size, avail_steps= steps[table_name], primary_keys= [])
        parts_ = [*parts, additional_main_part]
        ctes = [*steps[table_name], additional_cte]
        body = build_main_part(parts_).limit(size)
        body.set('with', exp.With(expressions = ctes))
        queries[table_name] = body.sql(dialect= dialect)
    return queries