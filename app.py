from flask import Flask, session, jsonify, request
import json
import os
from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time, outdated_packages_list
from scoring import score_model


# ##################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 


# ###################### Prediction Endpoint ######################
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Call the prediction function

    Returns:
        Return value for prediction outputs
    """
    data_file = request.files['input_file']
    predictions = model_predictions(data_file)
    return jsonify(predictions.tolist())


# ###################### Scoring Endpoint ######################
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score_check():
    """
    Check the score of the deployed model

    Returns:
        A single F1 score number
    """
    score = score_model(api=True)
    return jsonify(score)


# ###################### Summary Statistics Endpoint ######################
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Check means, medians, and modes for each column
    Returns:
        A list of all calculated summary statistics
    """
    statistic_info = dataframe_summary()

    return jsonify({
        'summary_stats': statistic_info
    })


# ###################### Diagnostics Endpoint ######################
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics_check():
    """
    Check timing and percent NA values

    Returns:
        Value for all diagnostics
    """
    timing_results = execution_time()
    missing_data_results = check_missing_data()
    dependency_check_results = outdated_packages_list()

    return jsonify({
        'timing': timing_results,
        'missing_data': missing_data_results,
        'dependency_check': dependency_check_results
    })


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
