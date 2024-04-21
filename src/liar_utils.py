"""This module contains utility classes and functions related to the
LIAR dataset"""

import pandas as pd
import os
from enum import Enum

class LIARLabel(Enum):

    PANTS_FIRE = 0
    FALSE = 1
    BARELY_TRUE = 2
    HALF_TRUE = 3
    MOSTLY_TRUE = 4
    TRUE = 5

LABEL_MAP = {
    "pants-fire": LIARLabel.PANTS_FIRE,
    "false": LIARLabel.FALSE,
    "barely-true": LIARLabel.BARELY_TRUE,
    "half-true": LIARLabel.HALF_TRUE,
    "mostly-true": LIARLabel.MOSTLY_TRUE,
    "true": LIARLabel.TRUE
}


def load_liar_dataset(data_folder="data/LIAR"):
    """Loads the LIAR dataset"""

    train = pd.read_csv(os.path.join(data_folder, "train.tsv"), sep="\t", index_col=0)
    val = pd.read_csv(os.path.join(data_folder, "valid.tsv"), sep="\t", index_col=0)
    test = pd.read_csv(os.path.join(data_folder, "test.tsv"), sep="\t", index_col=0)
    data = pd.concat([train, val, test])[["label", "statement"]]
    data["label"] = data["label"].apply(lambda x: LABEL_MAP[x].value)
    return data

    
def main():

    data = load_liar_dataset()
    print(data.head())

if __name__ == "__main__":
    main()
