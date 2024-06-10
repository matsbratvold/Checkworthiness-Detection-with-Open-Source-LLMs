"""This module contains utils related to the claimbuster dataset"""

from src.dataset_utils import convert_to_lora_dataset
from enum import Enum
from typing import List
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import sent_tokenize

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
    folder_path: str, 
    ncs_ratio=NCS_RATIO.TWO_AND_A_HALF, 
    use_binary_labels=True, 
    use_contextual_features=False,
    debate_transcripts_folder: str = None,
) -> pd.DataFrame:
    """Load the Claimbuser dataset

    Parameters
    ---------------------------
    folder_path: str
        The location where the dataset is located.
    ncs_ratio: NCS_RATION (default = NCS_RATIO.TWO_AND_A_HALF)
        The ratio of NCS to CFS sentences. This is only applied if
        use_binary_labels is set to True
    use_binary_labels: bool (default = True)
        If True, the updated dataset with labels CFS and NCS are used.
        If set to False, the original dataset with three classes is used
        UFS and NFS labels
    use_contextual_features (default = False)
        If set to True, the previous five sentences within the debate are added
        to the resulting dataframe
    debate_transcripts_folder (default = None)
        Location of the debate transcripts. Is only applied if 
        use_contextual_features is set to True
    """
    if use_binary_labels:
        data = pd.read_json(os.path.join(folder_path, f"{ncs_ratio.value}xNCS.json"))
        data.set_index("sentence_id", inplace=True)
        data = data.rename(columns={"label": "Verdict", "text": "Text"})
        if use_contextual_features:
            assert debate_transcripts_folder is not None, "You should provide a folder where the debate transcriptions are located"
            extractor = ClaimBusterContextualFeatureExtractor(debate_transcripts_folder)
            data = extractor.add_contextual_features(data)
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


class ClaimBusterContextualFeatureExtractor:
    """This class is used to extract contextual features from the original
    debate transcriptions"""

    def __init__(self, folder: str):
        self.all_sentences = []
        self.sentence_matcher = re.compile(r"[\.\?!]")
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), "r", encoding="utf8") as f:
                self.all_sentences += [(
                    file,
                    [line.strip() for line in sent_tokenize(f.read()) if len(line.strip()) > 0],
                )]

    def add_contextual_features(
        self, data: pd.DataFrame, output_path: str = None, sentences=5
    ) -> pd.DataFrame:
        """Add contextual features to the dataset based on the original debate transcriptions."""
        data["previous_sentences"] = data["Text"].apply(lambda x: self.find_previous_sentences(x, sentences))
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_json(output_path, orient="records")
        return data

    def find_previous_sentences(self, text: str, sentences: int) -> str:
        """Find the previous  within a debate transcript"""
        for _, lines in self.all_sentences:
            line: list
            for index, line in enumerate(lines):
                if index > 0 and text in line:
                    previous_sentences = lines[max(0, index - sentences) : index]
                    return " ".join(previous_sentences).replace("\n", "").strip()
        return ""


def main():
    folder_path = os.path.join("data", "ClaimBuster/datasets")
    data = load_claimbuster_dataset(folder_path)
    output_path = "data/ClaimBuster/datasets/lora.json"
    with open("prompts/ClaimBuster/standard/zero-shot-lora.txt") as f:
        instruction = f.read().replace("\n", " ").strip()
    lora = convert_to_lora_dataset(
        data=data, 
        instruction=instruction,
        output_path=output_path
    )
    print(lora.head())



if __name__ == "__main__":
    main()
