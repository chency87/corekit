import sys
import os
# Get the current directory (where your_script.py resides)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from src import *
import random, logging

logger = logging.getLogger('src.query')
metrics_logger =  logging.getLogger('metrics.query')



from src.db.db_manage import DBManager

sql = """SELECT `bond`.`bond_type` FROM `bond` INNER JOIN `molecule` ON `bond`.`molecule_id` = `molecule`.`molecule_id` WHERE `molecule`.`molecule_id` BETWEEN 'TR000' AND 'TR050'"""


# DBManager().export_sampledb('','', query= sql)

sql = "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1",
scm, r = DBManager().export_sampledb('./tests', database= 'california_schools.sqlite', query= sql, size= 10, to = './tests/db891_copy.sqlite')

# r = DBManager().export_database('./tests', database= 'db0.sqlite')
# print(scm)

# print(r)

print(len(r))
