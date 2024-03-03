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
    sep="\t"
  )
  dev = pd.read_csv(
    os.path.join(folder_path, "dataset_dev_v1_english.tsv"),
    sep="\t"
  )
  test = pd.read_csv(
    os.path.join(folder_path, "dataset_test_english.tsv"),
    sep="\t"
  )
  if merge_dataset:
    return pd.concat([train, dev, test])
  else:
    return [train, dev, test]