"""This module contains utils related to the claimbuster dataset"""

from enum import Enum
from typing import List
import pandas as pd
import os


class ClaimbusterMultiClassLabel(Enum):
    CFS = 1
    """ Check-worthy Factual Sentence """

    UFS = 0
    """ Unimportant Factual Sentence """

    NFS = -1
    """ Non-factual Sentence """


class ClaimbusterBinaryLabel(Enum):
    CFS = 1
    """ Check-worthy Factual Statement """

    NCS = 0
    """ Non-Check-worthy Sentence """

class ClaimbusterSpeakerParty(Enum):
    REPUBLICAN = "REPUBLICAN"
    DEMOCRAT = "DEMOCRAT"
    INDEPENDENT = "INDEPENDENT"

class ClaimBusterSpeakerTitle(Enum):
    GOVERNOR = "Governor"
    SENATOR = "Senator"
    PRESIDENT = "President"
    CONGRESSMAN = "Congressman"
    FORMER_VICE_PRESIDENT = "Former Vice President"
    INDEPENDENT_CANDIDATE = "Independent Candidate"
    VICE_PRESIDENT = "Vice President"


def load_claimbuster_dataset(folder_path, use_binary_labels=False) -> pd.DataFrame:
    """Load the Claimbuser dataset

    Parameters
    ---------------------------
    use_binary_labels: bool (default = False)
        If true, use the binary labels CFS and NCS, merging the original
        UFS and NFS labels
    """
    groundtruth_path = os.path.join(folder_path, "groundtruth.csv")
    groundtruth = pd.read_csv(groundtruth_path, index_col=0)
    crowdsourced_path = os.path.join(folder_path, "groundtruth.csv")
    crowdsourced = pd.read_csv(crowdsourced_path, index_col=0)
    data = pd.concat([groundtruth, crowdsourced])
    if use_binary_labels:
        data = merge_data_labels_into_binary(data)
    return data


def merge_data_labels_into_binary(data: pd.DataFrame) -> pd.DataFrame:
    """Merge the original mutliclass labels into the binary labels CFS and NCS"""
    replacements = {
        "Verdict": {
            ClaimbusterMultiClassLabel.CFS.value: ClaimbusterBinaryLabel.CFS.value,
            ClaimbusterMultiClassLabel.UFS.value: ClaimbusterBinaryLabel.NCS.value,
            ClaimbusterMultiClassLabel.NFS.value: ClaimbusterBinaryLabel.NCS.value,
        }
    }
    data = data.replace(replacements)
    return data

def filter_claimbuster_features(data: pd.DataFrame, features: List[str]):
    """Filters the dataset based on the provided features"""
    return data[features]

def main():
    folder_path = os.path.join("data", "ClaimBuster_Datasets/datasets")
    multi = load_claimbuster_dataset(folder_path)
    print(multi.head())
    binary = merge_data_labels_into_binary(multi)
    print(binary.head())


if __name__ == "__main__":
    main()
