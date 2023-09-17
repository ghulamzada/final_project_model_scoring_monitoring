import pickle
import pandas as pd
import subprocess
import timeit
import os
import json

# ################# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# ################# Model predictions #################
def model_predictions(test_data=test_data_path):
    """
    Read the deployed model and a test dataset, then calculate predictions

    Returns:
         A list containing all predictions
    """
    with open(prod_deployment_path + "/" + "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    test_data = pd.read_csv(test_data)

    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = test_data["exited"].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)
    return predicted


# ################# Summary statistics #################
def dataframe_summary():
    """
    Calculating summary statistics here
    Returns:
        A list or a dataframe containing all summary statistics
    """
    df = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', "exited"]
    df_statistic = pd.DataFrame(columns=['column_name', 'mean', 'median', 'standard_deviation'])

    statistic_list = []
    for col in columns:
        mean_value = df[col].mean()
        median_value = df[col].median()
        std_value = df[col].std()

        # Appending the values to the list
        statistic_list.append(col)
        statistic_list.append(mean_value)
        statistic_list.append(median_value)
        statistic_list.append(std_value)

        # Adding the statistic in a dataframe:
        # Creating a dictionary for the current column's statistics
        col_stats = {
            'column_name': col,
            'mean': mean_value,
            'median': median_value,
            'standard_deviation': std_value
        }
        # Appending the dictionary as a new row to the DataFrame
        df_statistic = df_statistic.append(col_stats, ignore_index=True)

    return statistic_list


# ################# Get timings #################
def check_missing_data():
    """
    Check for missing values in each column of dataframe and show their nans percentage
    Returns:
        A list of nans and nan-percentage in the dataframe
    """
    df = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    # The following code accomplishes all the data integrity checks:

    nas = list(df.isna().sum())

    # List comprehension to calculate the percent of each column that consists of NA values:
    nas_list = [(nas[i] / len(df.index)) for i in range(len(nas))]

    return nas_list


# ################# Get Timings #################
def execution_time():
    """
    Calculates timing of training.py and ingestion.py
    Returns:
        A list of 2 timing values in seconds
    """
    final_result = []

    start_time_ingestion = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ingestion = timeit.default_timer() - start_time_ingestion
    final_result.append(f"Time for ingestion in seconds: {timing_ingestion}")

    start_time_training = timeit.default_timer()
    os.system('python3 training.py')
    timing_training = timeit.default_timer() - start_time_training

    final_result.append(f"Time for training in seconds: {timing_training}")

    return final_result


# ################# Check dependencies #################
def outdated_packages_list():
    """
    Get a list of outdated packages
    Returns:
         Save the result as txt file
    """

    # Get all current installed packages in a "requirements" format.
    subprocess.check_call(['pip', 'list', '--outdated'],
                          stdout=open(prod_deployment_path + "/" + 'current_modul_versions.txt', 'w'))
    with open(prod_deployment_path + "/" + 'current_modul_versions.txt', 'r') as file:
        outdated_moduls = file.read()

    return outdated_moduls


if __name__ == '__main__':
    check_missing_data()
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
