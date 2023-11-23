
from utils.db_utils import get_db_conn
from loguru import logger
con = get_db_conn()

def query_algorithm_name():
    sql = f"select ALGORITHM_NAME from UMS_TSAD_ALGORITHM"
    res = con.execute(sql).fetchall()
    algorithm_name_list = [i[0] for i in res]
    logger.info(f'algorithm_name is {algorithm_name_list}')
    return algorithm_name_list

_ = query_algorithm_name()