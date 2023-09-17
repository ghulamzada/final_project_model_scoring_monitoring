import pickle
import os
import json

# ################# Load config.json and correct path variable #################
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

output_model_path = os.path.join(config['output_model_path'])

with open(output_model_path + "/" + "trainedmodel.pkl", 'rb') as file:
    model = pickle.load(file)

with open(output_model_path + "/" + "latestscore.txt", 'r') as file:
    f1score = file.read()

with open(dataset_csv_path + "/" + "ingestedfiles.txt", 'r') as file:
    ing_rec = file.read()


# ################### Deployment ###################
def store_model_into_pickle(trained_model=model, model_score= f1score, ingestion_records=ing_rec):
    """
   This function copies the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the
   deployment directory

    Args:
        trained_model: trained model
        model_score: f1 score from model scoring
        ingestion_records: records of ingestion process

    Returns:
         Saves trained model, model-score and ingested files
    """
    pickle.dump(trained_model, open(prod_deployment_path + "/" + "trainedmodel.pkl", 'wb'))

    with open(prod_deployment_path + "/" + "latestscore.txt", "w") as fl:
        fl.write(str(model_score))

    with open(prod_deployment_path + "/" + "ingestedfiles.txt", "w") as f:
        f.write(str(ingestion_records))


if __name__ == "__main__":
    store_model_into_pickle()
