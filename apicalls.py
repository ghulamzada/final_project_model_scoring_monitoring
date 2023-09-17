import requests
import json
import os


def create_api_calls():
    """
    This function make api calls

    Returns:
         A text file with all api call respones
    """

    # Specifying a URL that resolves the workspace
    URL = "http://127.0.0.1:8000"

    # Calling each API endpoint and store the responses
    files = {'input_file': ('testdata.csv', open('test_data/testdata.csv', 'rb'))}
    prediction_response = requests.post(URL + "/prediction", files=files)
    scoring_response = requests.get(URL + "/scoring")
    summarystats_response = requests.get(URL + "/summarystats")
    diagnostics_response = requests.get(URL + "/diagnostics")

    # Combining all API responses
    combined_outputs = {
        "prediction": prediction_response.json(),
        "scoring": scoring_response.json(),
        "summarystats": summarystats_response.json(),
        "diagnostics": diagnostics_response.json()
    }

    # Writing the responses to your workspace / to a file
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_model_path = os.path.join(config['output_model_path'])

    with open(output_model_path + "/" + "apireturns2.txt", "w") as file:
        file.write(json.dumps(combined_outputs))

    print("API outputs have been written to apireturns2.txt.")


if __name__ == "__main__":
    create_api_calls()
