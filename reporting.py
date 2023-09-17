import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from diagnostics import model_predictions


# ############## Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# ############# Reporting ############
def generate_confusion_matrix():
    """
    Calculates a confusion matrix using the test data and the deployed model
    Returns:
        Saves the confusion matrix to the workspace
    """
    predicted_values = model_predictions(test_data=test_data_path+"/"+"testdata.csv")
    actual_val = pd.read_csv(test_data_path + "/" + "testdata.csv")
    actual_values = actual_val["exited"]
    # Create a confusion matrix
    cm = confusion_matrix(actual_values, predicted_values)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the confusion matrix plot
    plt.savefig(output_model_path + "/" + 'confusionmatrix2.png')


if __name__ == '__main__':
    generate_confusion_matrix()
