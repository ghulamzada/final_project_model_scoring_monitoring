import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# ################ Load config.json and get path variables ################
with open('config.json', 'r') as f:
    config = json.load(f)

# dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")
output_model_score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
data = pd.read_csv(test_data_path)


# ################ Model Scoring ################
def score_model(api: bool = False, trained_model_path=model_path, test_data=data):
    """
    This function takes a trained model, load test data, and  calculates an F1 score for
    the model relative to test data.

    Returns:
        It writes the result in latestscore.txt file
    """
    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = test_data["exited"].values.reshape(-1, 1).ravel()

    with open(trained_model_path, 'rb') as file:
        model = pickle.load(file)

    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)

    if not api:
        with open(output_model_score_path, "w") as file:
            file.write(str([f1score]))
        return f1score

    if api:
        return f1score


if __name__ == "__main__":
    score_model()
