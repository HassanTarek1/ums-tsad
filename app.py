# Import necessary modules and functions
from flask import render_template, request
from factory import create_app
from settings import DevelopmentConfig
from loguru import logger
from dao.mdata.mdata import query_data_type, select_data_entity_by_status, select_inject_abn_types_by_data_entity, select_best_model_by_data_entity
from dao.mmodel.mmodel import query_algorithm_name
import numpy as np
import random
# Initialize the Flask application using a factory pattern for better scalability and organization
app = create_app()

# Define a route to handle requests to the root URL ('/')
@app.route('/', methods=['GET', 'POST'])
def index():
    # Handling GET request - typically used to serve the page with initial data
    if request.method == 'GET':
        # Querying data types (e.g., categories of datasets) from the database
        dataset_types = query_data_type()

        # Querying available algorithm names from the database - useful for populating dropdowns or lists in the UI
        algorithm_list = query_algorithm_name()

        # Querying information about data entities based on their status (e.g., active, inactive)
        data_entity_info_list = select_data_entity_by_status()

        # Flag to control the display of data in a tabular format on the frontend
        table_flag = False
        # Logging the list of data entities for debugging purposes
        logger.info(f'data_entity_info_list is {data_entity_info_list}')

        # Prepare a list to hold formatted data entities for rendering
        data_infos = []

        # Check if there are any data entities to display
        if len(data_entity_info_list) > 0:
            table_flag = True

            # Loop through each data entity and structure the data for frontend use
            for data_entity_info in data_entity_info_list:
                data_info = {'dataset_entity': data_entity_info[0], 'algorithms': data_entity_info[1],
                             'dataset_type': data_entity_info[2], 'inject_abn_types': data_entity_info[3],
                             'best_model': data_entity_info[4]}
                data_infos.append(data_info)

        # Render the 'index.html' template, passing in the queried data for use in the HTML template
        return render_template(template_name_or_list='index.html', dataset_types=dataset_types,
                               algorithm_list=algorithm_list, table_flag=table_flag, data_infos=data_infos)

    # Handling POST request - typically used for submitting form data to the server
    if request.method == 'POST':
        # This section can be expanded to handle form submissions.
        # For instance, processing user input or saving data to the database.
        return

# Check if the script is executed directly (not imported) and run the app
if __name__ == '__main__':
    # Starting the Flask application with configuration settings (like host and port) specified in DevelopmentConfig
    # This is typically used for local development and testing
    np.random.seed(42)
    random.seed(42)
    app.run(host=DevelopmentConfig.HOST, port=DevelopmentConfig.PORT)
