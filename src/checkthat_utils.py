""" This module contains utily functions and enum related to the 
CheckThat2021 Task 1a dataset containg tweets related to Covid.
"""

import enum
from typing import List, Union
import pandas as pd
import os

class CheckThatLabel(enum.Enum):
  CHECK_WORTHY = 1
  """ Check-worthy Tweet """

  NON_CHECK_WORTHY = 0
  """ Non-check-worthy Tweet """

def load_check_that_dataset(
    folder_path: str,
    merge_dataset: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
  """Loads the CheckThat2021 task 1a dataset"""
  train = pd.read_csv(
    os.path.join(folder_path, "dataset_train_v1_english.tsv"),
    index_col=1,
    sep="\t"
  )
  # Commenting out dev since it has a different structure.
  # dev = pd.read_csv(
  #   os.path.join(folder_path, "dataset_dev_v1_english.tsv"),
  #   index_col=1,
  #   sep="\t"
  # )
  test = pd.read_csv(
    os.path.join(folder_path, "dataset_test_english.tsv"),
    index_col=1,
    sep="\t"
  )
  if merge_dataset:
    combined = pd.concat([train, test])
    return combined
  else:
    return [train, test]