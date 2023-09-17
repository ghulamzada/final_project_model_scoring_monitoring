import ingestion
import pandas as pd
import numpy as np
import os
import json
import ast
from scoring import score_model
from training import train_model

# ################# Check and read new data #################
# first, read ingestedfiles.txt
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")


def check_read_new_data(prod_deploy_path=prod_deployment_path):
    """
    Determine whether the source data folder has files that aren't listed in ingestedfiles.txt.
    If new data found, ingestion process should proceed. otherwise, end the loop-process.

    Returns:
         Re-runs the ingestion process for new data
    """
    with open(prod_deploy_path + "/" + "ingestedfiles.txt", "r") as file:
        ingested_files = ast.literal_eval(file.read())

    files_not_list = 0

    for file in ingested_files:
        main_folder = os.listdir(os.getcwd() + "/" + input_folder_path)

        if file not in main_folder:
            print(f"There is files with same name in folder 'source_data'")
            files_not_list += 1

        # if you found new data, you should proceed. otherwise, do end the process here
        else:
            print(f"This file exists both in source_folder and ingestedfiles.txt: {file}")
            print(f"Stopping the process")

    if files_not_list > 0:
        print(f"Thus, starting ingestion:")
        print(f"Number of file that are new: {files_not_list}")
        ingestion.merge_multiple_dataframe()


# ################# Checking for model drift ################# check whether the score from the deployed model is
# different from the score from the model that uses the newest ingested data

def check_model_drift(prod_deploy_path=prod_deployment_path):
    """
    Check model drift and Perform the raw comparison test by checking whether the new score is worse than
    all previous scores.

    Returns:
        Re-runs all pipeline logic
    """
    model_path = prod_deploy_path + "/" + "trainedmodel.pkl"
    new_df = pd.read_csv(dataset_csv_path)
    model_drift = bool

    with open(prod_deploy_path + "/" + "latestscore.txt", "r") as ff:
        latest_scores = ast.literal_eval(ff.read())

    new_f1score = score_model(trained_model_path=model_path, test_data=new_df)

    # Perform the raw comparison test by checking whether the new score is worse than all previous scores:
    if new_f1score < np.min(latest_scores):
        model_drift = True

    elif new_f1score >= np.min(latest_scores):
        model_drift = False

    # ################# Deciding whether to proceed, part 2 #################
    # if you found model drift, you should proceed. otherwise, do end the process here
    print(f"New Model Score: {new_f1score}")
    print(f"Last Model Score: {latest_scores}")

    if model_drift:
        train_model(df=new_df)
        # if you found evidence for model drift, re-run the deployment.py script
        from deployment import store_model_into_pickle
        # run diagnostics.py and reporting.py for the re-deployed model
        from reporting import generate_confusion_matrix
        from apicalls import create_api_calls
        store_model_into_pickle()
        generate_confusion_matrix()
        create_api_calls()
        print("FINISHED EXECUTING 'fullprocess.py")


if __name__ == "__main__":
    check_read_new_data()
    check_model_drift()
