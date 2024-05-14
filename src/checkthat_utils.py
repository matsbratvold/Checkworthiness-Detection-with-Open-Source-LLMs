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

def main():
    folder_path = "data/CheckThat"
    data = load_check_that_dataset(folder_path)
    data = data[["check_worthiness", "tweet_text"]]
    # Rename columns
    data = data.rename(columns={"check_worthiness": "label", "tweet_text": "text"})
    print(data.head())
    save_path = "/cluster/home/matssbra/fake-news-detection/Fake-news-detection/claimbuster-spotter-master/svm"
    data.to_json(os.path.join(save_path, "checkthat_dataset.json"), orient="records")
    # output_path = "data/CheckThat/lora.json"
    # with open("prompts/CheckThat/standard/zero-shot-lora.txt") as f:
    #     instruction = f.read().replace("\n", " ").strip()
    # print(data.columns)
    # lora = convert_to_lora_dataset(
    #     data=data, 
    #     instruction=instruction,
    #     output_path=output_path,
    #     text_column="tweet_text",
    #     label_column="check_worthiness",
    # )
    # print(lora.head())



if __name__ == "__main__":
    main()