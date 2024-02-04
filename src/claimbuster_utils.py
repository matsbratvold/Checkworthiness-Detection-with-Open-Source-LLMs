"""This module contains utils related to the claimbuster dataset"""

from enum import Enum
from typing import List
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer


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


class NCS_RATIO(Enum):
    """Ratio between CFS and NCS"""

    DOUBLE = 2
    TWO_AND_A_HALF = 2.5
    TRIPLE = 3


def load_claimbuster_dataset(
    folder_path: str, ncs_ratio=NCS_RATIO.TWO_AND_A_HALF, use_binary_labels=True
) -> pd.DataFrame:
    """Load the Claimbuser dataset

    Parameters
    ---------------------------
    use_binary_labels: bool (default = False)
        If true, use the binary labels CFS and NCS, merging the original
        UFS and NFS labels
    """
    if use_binary_labels:
        data = pd.read_json(os.path.join(folder_path, f"{ncs_ratio.value}xNCS.json"))
        data.set_index("sentence_id", inplace=True)
        data = data.rename(columns={"label": "Verdict", "text": "Text"})
        return data
    crowdsourced = pd.read_csv(
        os.path.join(folder_path, "crowdsourced.csv"), index_col=0
    )
    groundtruth = pd.read_csv(os.path.join(folder_path, "groundtruth.csv"), index_col=0)
    return pd.concat([crowdsourced, groundtruth])


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


def extract_tfid_features(data: pd.DataFrame, vectorizer: TfidfVectorizer, fit=False):
    """Extract tfid features from the dataset"""
    if fit:
        vectorizer.fit(data["Text"])
    tfidf_values = vectorizer.transform(data["Text"])
    tfidf_feature_names = vectorizer.get_feature_names_out()
    tfidf_data = pd.DataFrame(
        tfidf_values.toarray(), columns=tfidf_feature_names, index=data.index
    )
    return pd.concat([data, tfidf_data], axis=1)


def main():
    folder_path = os.path.join("data", "ClaimBuster_Datasets/datasets")
    multi = load_claimbuster_dataset(folder_path)
    print(multi.head())
    binary = merge_data_labels_into_binary(multi)
    print(binary.head())


if __name__ == "__main__":
    main()
