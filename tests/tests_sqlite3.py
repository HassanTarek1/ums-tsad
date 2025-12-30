import sqlite3 as sl
con = sl.connect('/db/tsad.db')


# create table data
# with con:
#     con.execute("""
#         CREATE TABLE IF NOT EXISTS UMS_TSAD_DATA (
#             DATA_ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
#             DATA_ENTITY TEXT,
#             DATA_DESC TEXT,
#             DATASET_TYPE TEXT,
#
#             DATA_STATUS INTEGER
#         );
#     """)

# insert data to table data

from model_trainer.entities import ANOMALY_ARCHIVE_ENTITIES,MACHINES,SMAP_CHANNELS


# insert ANOMALY_ARCHIVE_ENTITIES
# for data_entity in ANOMALY_ARCHIVE_ENTITIES:
#     print(data_entity)
#     with con:
#         con.execute("""
#             INSERT INTO UMS_TSAD_DATA (DATA_ENTITY,DATASET_TYPE,DATA_STATUS) VALUES (?,?,?);
#         """,(data_entity,'anomaly_archive',0))

# for data_entity in MACHINES:
#     print(data_entity)
#     with con:
#         con.execute("""
#             INSERT INTO UMS_TSAD_DATA (DATA_ENTITY,DATASET_TYPE,DATA_STATUS) VALUES (?,?,?);
#         """,(data_entity,'smd',0))






# create table algorithm
# with con:
#     con.execute("""
#         CREATE TABLE IF NOT EXISTS UMS_TSAD_ALGORITHM (
#             ALGORITHM_ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
#             ALGORITHM_NAME TEXT,
#             ALGORITHM_PARA TEXT,
#             ALGORITHM_DESC TEXT
#         );
#     """)

# insert data to table algorithm
# ALGORITHM = ['DGHL', 'RNN', 'LSTMVAE', 'NN', 'MD', 'RM']
# ALGORITHM = ['LOF']
# ALGORITHM = ['ABOD','CBLOF','CD','COF','COPOD','ECOD','HBOS','IForest','INNE','KDE']
# for algorithm in ALGORITHM:
#     print(algorithm)
#     with con:
#         con.execute("""
#             INSERT INTO UMS_TSAD_ALGORITHM (ALGORITHM_NAME,ALGORITHM_PARA,ALGORITHM_DESC) VALUES (?,?,?);
#         """,(algorithm,'null','null'))

# drop table
# with con:
#     con.execute("""
#         DROP TABLE UMS_TSAD_DATA;
#     """)

delete_list = ['ABOD','CBLOF','CD','COF','COPOD','ECOD','HBOS','IForest','INNE','KDE']
# delete row
with con:
    con.execute("""
    DELETE FROM UMS_TSAD_ALGORITHM
    WHERE ALGORITHM_NAME='INNE';

    """)



# with con:
#     con.execute("""
#         ALTER TABLE UMS_TSAD_DATA ADD COLUMN BEST_MODEL TEXT;
#     """)



con.commit()