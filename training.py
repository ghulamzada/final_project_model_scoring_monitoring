import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

# ################## Load config.json and get path variables ##################
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv")
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")

data = pd.read_csv(dataset_csv_path)


# ################ Model Training ################
def train_model(df=data):
    """
    This functions trains a logistic regression model

    Returns:
        trained model and saves is as pickle file
    """

    # Logistic regression for training
    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = df["exited"].values.reshape(-1, 1).ravel()

    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                 intercept_scaling=1, l1_ratio=None, max_iter=100,
                                 multi_class='auto', n_jobs=None, penalty='l2',
                                 random_state=42, solver='liblinear', tol=0.0001, verbose=0,
                                 warm_start=False)

    # Fitting the logistic regression
    model = log_reg.fit(X, y)
    # Writing the trained model to my workspace as trainedmodel.pkl
    pickle.dump(model, open(model_path, 'wb'))


if __name__ == "__main__":
    train_model()
