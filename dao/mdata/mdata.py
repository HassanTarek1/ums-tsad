
from utils.db_utils import get_db_conn
from loguru import logger
con = get_db_conn()


def query_data_type():
    sql = "select distinct DATASET_TYPE from UMS_TSAD_DATA"
    res = con.execute(sql).fetchall()

    dataset_types = [i[0] for i in res]
    logger.info(f'dataset_type is {dataset_types}')
    return dataset_types

# query_data_type()

def query_data_by_type(_data_type):
    sql = f"select DATA_ENTITY from UMS_TSAD_DATA where DATASET_TYPE = '{_data_type}'"
    res = con.execute(sql).fetchall()
    data_entity_list = [i[0] for i in res]
    logger.info(f'data_entity_list is {data_entity_list}')
    return data_entity_list

# _data_type = 'anomaly_archive'
# query_data_by_type(_data_type)



# _data_type = 'anomaly_archive'
# query_data_by_type(_data_type)

def update_data_status_by_name(_data_name,_status=1):
    sql = f"update UMS_TSAD_DATA set DATA_STATUS = ?  where DATA_ENTITY = ?"
    con.execute(sql,(_status,_data_name))
    con.commit()


# _data_name = '125_UCR_Anomaly_ECG4'
# update_data_status_by_name(_data_name)

def update_data_algorithm_by_name(_data_name,_algorithm):
    sql = f"update UMS_TSAD_DATA set ALGORITHM_NAME = ?  where DATA_ENTITY = ?"
    con.execute(sql,(_algorithm,_data_name))
    con.commit()

# _data_name = '125_UCR_Anomaly_ECG4'
# _algorithm = 'DGHL_RNN'
# update_data_algorithm_by_name(_data_name,_algorithm)

def select_data_entity_by_status(_status = 1):
    sql = f"select DATA_ENTITY,ALGORITHM_NAME,DATASET_TYPE,INJECT_ABN_TYPES,BEST_MODEL from UMS_TSAD_DATA where DATA_STATUS >= '{_status}'"
    data_entity_info_list = con.execute(sql).fetchall()

    logger.info(f'data_entity_info_list is {data_entity_info_list}')
    return data_entity_info_list

# select_data_entity_by_status(_status = 1)

def select_data_status_by_entity(_data_name = None):
    sql = f"select DATA_STATUS from UMS_TSAD_DATA where DATA_ENTITY = '{_data_name}'"
    data_status = con.execute(sql).fetchone()[0]

    logger.info(f'data_status is {data_status}')
    return data_status

# _data_name = '125_UCR_Anomaly_ECG4'
# select_data_status_by_entity(_data_name = _data_name)

def update_data_inject_abn_types_by_name(_data_name,_inject_abn_types):
    sql = f"update UMS_TSAD_DATA set INJECT_ABN_TYPES = ?  where DATA_ENTITY = ?"
    con.execute(sql,(_inject_abn_types,_data_name))
    con.commit()

# _data_name = '125_UCR_Anomaly_ECG4'
# _inject_abn_types = 'spikes_contextual'
# update_data_inject_abn_types_by_name(_data_name,_inject_abn_types)


def select_inject_abn_types_by_data_entity(_data_name = None):
    sql = f"select INJECT_ABN_TYPES from UMS_TSAD_DATA where DATA_ENTITY = '{_data_name}'"
    inject_abn_type_list = con.execute(sql).fetchall()
    inject_abn_types = inject_abn_type_list[0][0]
    logger.info(f'inject_abn_types is {inject_abn_types}')
    return inject_abn_types

# _data_name = '104_UCR_Anomaly_NOISEapneaecg4'
# select_inject_abn_types_by_data_entity(_data_name = _data_name)


def select_algorithms_by_data_entity(_data_name):
    sql = f"select ALGORITHM_NAME from UMS_TSAD_DATA where DATA_ENTITY = '{_data_name}'"
    algorithms = con.execute(sql).fetchone()[0]

    logger.info(f'algorithms is {algorithms}')
    return algorithms

# _data_name = '125_UCR_Anomaly_ECG4'
# select_algorithms_by_data_entity(_data_name)


def update_data_best_model_by_name(_data_name,_best_model):
    sql = f"update UMS_TSAD_DATA set BEST_MODEL = ?  where DATA_ENTITY = ?"
    con.execute(sql,(_best_model,_data_name))
    con.commit()


# _data_name = '125_UCR_Anomaly_ECG4'
# _best_model = 'RNN_1'
# update_data_best_model_by_name(_data_name,_best_model)


def select_best_model_by_data_entity(_data_name):
    sql = f"select BEST_MODEL from UMS_TSAD_DATA where DATA_ENTITY = '{_data_name}'"
    algorithms = con.execute(sql).fetchone()[0]

    logger.info(f'algorithms is {algorithms}')
    return algorithms

# _data_name = '125_UCR_Anomaly_ECG4'
# select_best_model_by_data_entity(_data_name=_data_name)


def delete_data_by_data_entity(_data_name):
    sql = f"delete from UMS_TSAD_DATA where DATA_ENTITY = '{_data_name}'"
    con.execute(sql)
    con.commit()

_data_name = '233_UCR_Anomaly_mit14157longtermecg'
delete_data_by_data_entity(_data_name)