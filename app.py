from flask import render_template, request
from loguru import logger

from dao.mdata.mdata import add_new_datasets
from dao.mdata.mdata import check_new_dataset
from dao.mdata.mdata import query_data_type, select_data_entity_by_status
from dao.mmodel.mmodel import query_algorithm_name
from factory import create_app
from settings import DevelopmentConfig
from utils.utils import get_args_from_cmdline

app = create_app()
app.secret_key = 'secret_key'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        args = get_args_from_cmdline()
        data_dir = args['dataset_path']
        new_datasets = check_new_dataset(data_dir)
        new_datasets_path = add_new_datasets(new_datasets, data_dir)
        dataset_types = query_data_type()

        algorithm_list = query_algorithm_name()

        data_entity_info_list = select_data_entity_by_status()

        table_flag = False
        logger.info(f'data_entity_info_list is {data_entity_info_list}')
        data_infos = []
        if len(data_entity_info_list) > 0:
            table_flag = True

            data_infos = []
            for data_entity_info in data_entity_info_list:
                data_info = {'dataset_entity': data_entity_info[0], 'algorithms': data_entity_info[1],
                             'dataset_type': data_entity_info[2], 'inject_abn_types': data_entity_info[3],
                             'best_model': data_entity_info[4]}
                data_infos.append(data_info)
        return render_template(template_name_or_list='index.html', data_dir=data_dir, dataset_types=dataset_types,
                               algorithm_list=algorithm_list, table_flag=table_flag, data_infos=data_infos)

    if request.method == 'POST':
        return


if __name__ == '__main__':
    app.run(host=DevelopmentConfig.HOST, port=DevelopmentConfig.PORT)
