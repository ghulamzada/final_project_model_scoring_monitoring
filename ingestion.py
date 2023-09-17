import pandas as pd
import os
import json
from datetime import datetime

# ############ Load config.json and get input and output paths ############
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


date_time_obj = datetime.now()
current_time = str(date_time_obj.year) + "/" + str(date_time_obj.month) + "/" + str(date_time_obj.day)


# ############ Data ingestion ############
def merge_multiple_dataframe(output_location_ingestion_record: str = "ingestedfiles.txt",
                             output_location_merged_dataframe: str = "finaldata.csv",
                             file_format: str = ".csv") -> pd.DataFrame:
    """
    Check for datasets, compile them together, and write to an output file

    Returns:
        A single dataframe
    """

    # Creating an empty dataframe
    final_dataframe = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity',
                                            'number_of_employees', 'exited'])

    # Variable needed for processing
    current_repo = os.listdir(os.getcwd())
    ingestion_record = []

    # Looping through main-folder / repo
    for directory in current_repo:
        if directory == input_folder_path:
            filenames = os.listdir(os.getcwd() + "/" + directory)
            # Looping for file names in a specific folder
            for each_filename in filenames:
                # Searching file in CSV-Format
                if each_filename.endswith(file_format):
                    # Adding of CSV-File name for ingestion recording
                    ingestion_record.append(each_filename)
                    current_df = pd.read_csv(os.getcwd() + "/" + directory + "/" + each_filename)
                    final_dataframe = final_dataframe.append(current_df).reset_index(drop=True)

    # Save final dataframe
    final_dataframe.drop_duplicates(inplace=True)
    final_dataframe.to_csv(output_folder_path + "/" + output_location_merged_dataframe, index=False)

    # Save ingestion records
    with open(output_folder_path + "/" + output_location_ingestion_record, 'w') as my_file:
        my_file.write(str(ingestion_record))

    return final_dataframe


if __name__ == '__main__':
    merge_multiple_dataframe()
