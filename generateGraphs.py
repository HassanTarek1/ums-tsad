import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path_new = "D:\Education\Master's courses\Semester 3\Research project\my docs/results/combinations/synthetic contextual.csv"

if __name__ == '__main__':

    # First, we'll try reading without specifying an encoding to see if it's in UTF-8
    try:
        data_new = pd.read_csv(file_path_new)
        # If successful, show the first few rows to understand its structure
        first_rows_new = data_new.head()
    except Exception as e:
        # If there's an issue, we'll attempt with 'ISO-8859-1' encoding
        try:
            data_new = pd.read_csv(file_path_new, encoding='ISO-8859-1')
            first_rows_new = data_new.head()
        except Exception as e:
            # If still an issue, return the exception
            first_rows_new = str(e)

    heatmap_data = pd.DataFrame()

    num_tables = int(len(data_new.columns) / 2)

    # Split the data into separate DataFrames for each table
    tables = {}
    for i in range(num_tables):
        # Get the column indices for the two columns of each table
        col1 = 2 * i
        col2 = 2 * i + 1
        # Assuming the first row contains header information, which we skip
        tables[data_new.columns[col1]] = data_new.iloc[1:, col1:col2 + 1].set_index(data_new.columns[col1])

    # Verify the first table's data as an example
    genesis = tables[list(tables.keys())[0]]
    anomaly_archive = tables[list(tables.keys())[1]]
    skab10 = tables[list(tables.keys())[2]]
    skab23 = tables[list(tables.keys())[3]]
    smd = tables[list(tables.keys())[4]]

    # Convert the scores to numeric values for plotting
    genesis[data_new.columns[1]] = pd.to_numeric(genesis[data_new.columns[1]], errors='coerce')
    anomaly_archive[data_new.columns[3]] = pd.to_numeric(anomaly_archive[data_new.columns[3]], errors='coerce')
    skab10[data_new.columns[5]] = pd.to_numeric(skab10[data_new.columns[5]], errors='coerce')
    skab23[data_new.columns[7]] = pd.to_numeric(skab23[data_new.columns[7]], errors='coerce')
    skab23[data_new.columns[7]] = pd.to_numeric(skab23[data_new.columns[7]], errors='coerce')
    smd[data_new.columns[9]] = pd.to_numeric(smd[data_new.columns[9]], errors='coerce')



    genesis = pd.DataFrame(genesis)
    genesis.reset_index(inplace=True)
    genesis.dropna(inplace=True)
    genesis.rename(columns={"Genesis Anomaly label": "algorithm", "Unnamed: 1": "f1-score"},inplace=True)
    genesis['BaseAlgorithm'] = genesis['algorithm'].str.extract(r'(\D+)', expand=False)

    anomaly_archive = pd.DataFrame(anomaly_archive)
    anomaly_archive.reset_index(inplace=True)
    anomaly_archive.dropna(inplace=True)
    anomaly_archive.rename(columns={"Anomaly Archive 001": "algorithm", "Unnamed: 3": "f1-score"}, inplace=True)
    anomaly_archive['BaseAlgorithm'] = anomaly_archive['algorithm'].str.extract(r'(\D+)', expand=False)

    skab10 = pd.DataFrame(skab10)
    skab10.reset_index(inplace=True)
    skab10.dropna(inplace=True)
    skab10.rename(columns={"Skab valve 1-0": "algorithm", "Unnamed: 5": "f1-score"}, inplace=True)
    skab10['BaseAlgorithm'] = skab10['algorithm'].str.extract(r'(\D+)', expand=False)

    skab23 = pd.DataFrame(skab23)
    skab23.reset_index(inplace=True)
    skab23.dropna(inplace=True)
    skab23.rename(columns={"Skab valve 2-3": "algorithm", "Unnamed: 7": "f1-score"}, inplace=True)
    skab23['BaseAlgorithm'] = skab23['algorithm'].str.extract(r'(\D+)', expand=False)

    smd = pd.DataFrame(smd)
    smd.reset_index(inplace=True)
    smd.dropna(inplace=True)
    smd.rename(columns={"smad machine 1-6": "algorithm", "Unnamed: 9": "f1-score"}, inplace=True)
    smd['BaseAlgorithm'] = smd['algorithm'].str.extract(r'(\D+)', expand=False)

    print(smd.head())

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    sns.barplot(x='BaseAlgorithm', y='f1-score', data=genesis, errorbar=None, hue='BaseAlgorithm',legend=False, ax=axs[0, 0])
    axs[0, 0].set_title('Genesis Anomaly labels', color='blue', fontweight='bold', fontsize=14)
    sns.barplot(x='BaseAlgorithm', y='f1-score', data=anomaly_archive, errorbar=None, hue='BaseAlgorithm',legend=False, ax=axs[0, 1])
    axs[0, 1].set_title('anomaly_archive 001', color='blue', fontweight='bold', fontsize=14)
    sns.barplot(x='BaseAlgorithm', y='f1-score', data=skab10, errorbar=None, hue='BaseAlgorithm',legend=False, ax=axs[1, 0])
    axs[1, 0].set_title('Skab valve1-0', color='blue', fontweight='bold', fontsize=14)
    sns.barplot(x='BaseAlgorithm', y='f1-score', data=skab23, errorbar=None, hue='BaseAlgorithm',legend=False, ax=axs[1, 1])
    axs[1, 1].set_title('Skab valve2-3', color='blue', fontweight='bold', fontsize=14)
    sns.barplot(x='BaseAlgorithm', y='f1-score', data=smd, errorbar=None, hue='BaseAlgorithm',legend=False, ax=axs[2, 0])
    axs[2, 0].set_title('SMD machine1-6', color='blue', fontweight='bold', fontsize=14)


    axs[2, 1].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Display the figure
    plt.show()




    # Drop the first row which contains headers from the CSV
    heatmap_data_dict = {}

    # Iterating through each dataset to fill the dictionary
    for dataset, df in tables.items():
        for index, row in df.iterrows():
            if index not in heatmap_data_dict:
                heatmap_data_dict[index] = {}
            heatmap_data_dict[index][dataset] = row.values[0]

    # Convert the dictionary to a DataFrame
    heatmap_df = pd.DataFrame.from_dict(heatmap_data_dict, orient='index')

    # Now let's make sure all data is numeric
    heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce')

    # We'll use seaborn to create the heatmap
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)
    # plt.title('Combined Heatmap of Algorithm Performance')
    # plt.ylabel('Algorithm')
    # plt.xlabel('Dataset')
    # plt.xticks(rotation=45, ha='right')
    #
    # plt.show()
    # plt.figure(figsize=(12, 6))
    # heatmap_df.plot(kind='line', marker='o')
    # plt.title('Line Chart Across Datasets')
    # plt.xlabel('Algorithm')
    # plt.ylabel('Score')
    # plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.xticks(rotation=90, ticks=range(len(heatmap_df.index)), labels=heatmap_df.index)
    # plt.tight_layout()
    # plt.show()
    #
    # # Clustered bar chart for the first two datasets
    # plt.figure(figsize=(10, 8))
    # heatmap_df[list(heatmap_df.columns)[:2]].plot(kind='bar', edgecolor='black')
    # plt.title('Clustered Bar Chart for First Two Datasets')
    # plt.xlabel('Algorithm')
    # plt.ylabel('Score')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

    # sns.pairplot(heatmap_df)
    # plt.show()
