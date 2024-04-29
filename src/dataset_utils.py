"""This module contains common utils to be used for several datasets"""

import enum
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import torch

class CustomDataset(enum.Enum):
    CLAIMBUSTER = "ClaimBuster"
    CHECK_THAT = "CheckThat"
    LIAR = "LIAR"
    RAWFC = "RAWFC"

class ProgressDataset(torch.utils.data.Dataset):
    """Dataset that can be used with tdqm to show progress"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

def convert_to_lora_dataset(
    data: pd.DataFrame, 
    instruction:str, 
    label_column="Verdict",
    text_column="Text",
    output_path: str = None
):
    """Convert the dataset to a format that can be used to fine-tune LLMs
    using the LORA technique."""
    lora_data = pd.DataFrame(
        index=data.index,
        data={
            "text": data.apply(lambda row: f"""<s>[INST] {instruction} '''{row[text_column]}''' [\INST] \\n {row[label_column]} </s>""", 1),
            "instruction": [instruction for _ in range(len(data))],
            "input": data[text_column],
            "output": data[label_column],
        },
    )
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lora_data.to_json(output_path, orient="records")
    return lora_data

def generate_cross_validation_datasets(
        data: pd.DataFrame, 
        n_splits=4, 
        label_column="Verdict",
        folder_path=None
    ):
    """Splits a dataset into n_splits dataset that is used for cross validation."""
    splitter = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    datasets = []
    existing_splits = set(os.listdir(folder_path))
    existing_train = [file for file in existing_splits if "train" in file]
    existing_test = [file for file in existing_splits if "test" in file]
    if len(existing_train) == n_splits and len(existing_test) == n_splits:
        for i in range(n_splits):
            train = pd.read_json(os.path.join(folder_path, f"train_{i}.json"))
            test = pd.read_json(os.path.join(folder_path, f"test_{i}.json"))
            datasets.append((train, test))
        return datasets
    for i, (train_index, test_index) in enumerate(splitter.split(data, data[label_column])):
        train = data.iloc[train_index].copy()
        test = data.iloc[test_index].copy()
        if folder_path is not None:
            os.makedirs(folder_path, exist_ok=True)
            train[train.index.name] = train.index
            train.to_json(os.path.join(folder_path, f"train_{i}.json"), orient="records")
            test[test.index.name] = test.index
            test.to_json(os.path.join(folder_path, f"test_{i}.json"), orient="records")
        datasets.append((train, test))
    return datasets
def main():
    pass
if __name__ == "__main__":
    main()