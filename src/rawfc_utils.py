"""This module contains utily functions and enums related to the RAWFC dataset
from Yang et al. 2022 (https://aclanthology.org/2022.coling-1.230)"""

import enum
import os
import pandas as pd

class RAWFCLabel(enum.Enum):

    FALSE = 0
    HALF_TRUE = 1
    TRUE = 2

LABEL_MAP = {
    "false": RAWFCLabel.FALSE,
    "half": RAWFCLabel.HALF_TRUE,
    "true": RAWFCLabel.TRUE
}

def label_to_name(label: RAWFCLabel):
    label_to_name = {
        RAWFCLabel.FALSE: "False",
        RAWFCLabel.HALF_TRUE: "Half true",
        RAWFCLabel.TRUE: "True"
    }
    return label_to_name[label]

def generate_dataset_from_individual_reports(folder_path: str):
    """Generate a dataset from individual reports. It assumes that the 
    reports are split into the following folders; train, val, and test"""
    data_frame = pd.DataFrame()
    for folder in ["train", "val", "test"]:
        individual_files = os.listdir(os.path.join(folder_path, folder))
        for file in individual_files:
            data = pd.DataFrame(
                data=pd.read_json(os.path.join(folder_path, folder, file)),
                columns=["event_id", "claim", "label"]
            )
            data.set_index("event_id", inplace=True)
            data["label"] = data["label"].apply(lambda x: LABEL_MAP[x].value)
            # remove duplicates
            data.drop_duplicates(inplace=True)
            data_frame = pd.concat([data_frame, data])
    data_frame.to_csv(os.path.join(folder_path, "data.csv"))
    return data_frame

def load_rawfc_dataset(folder_path: str):
    """Loads the RAWFC dataset. If it has not been generated get, then 
    it will be generated from the individual reports"""
    path = os.path.join(folder_path, "data.csv")
    if not os.path.exists(path):
        return generate_dataset_from_individual_reports(folder_path)
    return pd.read_csv(path, index_col=0)

def main():
    data = load_rawfc_dataset("data/RAWFC")
    print(data.head())

if __name__ == "__main__":
    main()

