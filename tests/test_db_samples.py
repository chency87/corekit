import sys
import os
# Get the current directory (where your_script.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from typing import List
from src import *
from sqlglot import parse_one
import random, logging
from src.db.sample import sample_small_db, sample_data_from_original_db
context = get_ctx()


logger = logging.getLogger('src.db.samples')


# def test_generate_schema_query(original_host_or_path, original_db, queries):
    


def test_sample_samll_database(original_host_or_path, original_db,  queries: List[str], to_host_or_path, to_database):

    schema , inserts = DBManager().sample(queries, original_host_or_path, original_database= original_db, to_host_or_path= to_host_or_path, to_database= to_database)

    logger.info(schema)

    logger.info(inserts)

    # ddls = DBManager().get_schema(original_host_or_path, database= original_db)
    # schemas, steps = sample(ddls, queries)
    # logger.info(schemas)
    # logger.info(steps)

    # DBManager().create_database(schemas, stmts= [], host_or_path= '', database= '')

def test_predicates_extraction():
    schema = DBManager().get_schema('./tests', 'financial.sqlite')

    # schema = jsonify_ddl(schema)
    sql = """SELECT DISTINCT T2.account_id FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.k_symbol = 'SIPO' AND T3.A2 = 'Pisek'"""
    new_scm, samples = sample_data_from_original_db(schema, sql)
    for q, v in samples.items():
        logger.info(str(v))

def test_sample_small_db(original_host_or_path, original_db,  queries: List[str], to_host_or_path, to_database):

    scm, inserts = sample_small_db(queries, original_host_or_path= original_host_or_path, original_database= original_db, to_host_or_path= to_host_or_path, to_database= to_database)

    # logger.info(inserts)
    # logger.info()




# # sql = """SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1"""
# sql = """SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3"""
# # sql = """SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode where T1.CDSCode = '200' ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1"""

# sql = """SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18"""

# # print(repr(parse_one(sql, dialect = 'sqlite')))
sql = """SELECT DISTINCT T2.account_id FROM trans AS T1 INNER JOIN account AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T1.k_symbol = 'SIPO' AND T3.A2 = 'Pisek'"""
# test_sample_samll_database('./tests', 'financial.sqlite', sql, to_host_or_path= './tests', to_database= 'db_142.sqlite')

# test_predicates_extraction()

test_sample_small_db('./tests', 'financial.sqlite', sql, to_host_or_path= './tests', to_database= 'db_142.sqlite')


# insert_stmt = f"INSERT INTO `{table_name}` ({columns}) VALUES ({values});"
# print(repr(parse_one("INSERT INTO tbl (col1, col2, col3) values (1, 2, 3)")))


# Insert(
#   this=Schema(
#     this=Table(
#       this=Identifier(this=tbl, quoted=False)),
#     expressions=[
#       Identifier(this=col1, quoted=False),
#       Identifier(this=col2, quoted=False),
#       Identifier(this=col3, quoted=False)]),
#   expression=Values(
#     expressions=[
#       Tuple(
#         expressions=[
#           Literal(this=1, is_string=False),
#           Literal(this=2, is_string=False),
#           Literal(this=3, is_string=False)])]))