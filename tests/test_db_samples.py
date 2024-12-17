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

context = get_ctx()


logger = logging.getLogger('src.db.samples')


def test_sample_samll_database(original_host_or_path, original_db,  queries: List[str], to_host_or_path, to_database):

    DBManager().sample(queries, original_host_or_path, original_database= original_db, to_host_or_path= './tests', to_database= 'small_california_schools.sqlite')
    # ddls = DBManager().get_schema(original_host_or_path, database= original_db)
    # schemas, steps = sample(ddls, queries)
    # logger.info(schemas)
    # logger.info(steps)

    # DBManager().create_database(schemas, stmts= [], host_or_path= '', database= '')



# sql = """SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1"""
sql = """SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3"""
# sql = """SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode where T1.CDSCode = '200' ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1"""

sql = """SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18"""

# print(repr(parse_one(sql, dialect = 'sqlite')))
test_sample_samll_database('./tests', 'california_schools.sqlite', sql, to_host_or_path= None, to_database= None)