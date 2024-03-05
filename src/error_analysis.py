"""This module contains functions used for error analysis,
including exploration of false positives and false negatives"""

from typing import Iterable
import pandas as pd
from llm import HuggingFaceModel
import enum
import os

class Dataset(enum.Enum):

    CLAIMBUSTER = "ClaimBuster"
    CHECK_THAT = "CheckThat"

def print_padded_text(text: str, total_length = 50):
    """Pad with hastags at the beggining and end of the text and center it."""
    padding_length_left = (total_length-len(text))//2
    padding_length_right = total_length - len(text) - padding_length_left
    print(f"{'#': <{padding_length_left}}{text}{'#': >{padding_length_right}}")

def generate_error_analysis_report(
    results: Iterable[pd.DataFrame],
    models: Iterable[HuggingFaceModel],
    folder_path: str = None,
    label_column_name = "Verdict",
    text_column_name = "Text",
):
    """Generate an error report for the given results and models, looking at
    the number of false positive and false negatives. Set folder_path to save 
    the resulting dataframes."""
    false_positive_ids = set()
    false_negative_ids = set()
    overlapping_false_positive_ids = None
    overlapping_false_negative_ids = None
    model_names = [model.name for model in models]
    filenames = ["false_negatives.csv", "false_positives.csv"]
    for name, result in zip(model_names, results):
        print("#" * 50)
        print_padded_text(name)
        # Print empty predictions
        empty_prediction_count = len(result[result["score"].isna()])
        wrong_format_count = len(result[result["reasoning"].map(lambda x: x is None or "####" in str(x))])
        false_negatives = result[result.apply(lambda x: x[label_column_name] == 1 and x["score"] < 30, axis=1)]
        false_positives = result[result.apply(lambda x: x[label_column_name] == 0 and x["score"] > 70, axis=1)]
        print_padded_text(f"False positives: {len(false_positives)}")
        print_padded_text(f"False negatives: {len(false_negatives)}")
        print_padded_text(f"Empty predictions: {empty_prediction_count}")
        print_padded_text(f"Wrong output format: {wrong_format_count}")
        if folder_path is not None:
            for filename, data in zip(filenames, [false_negatives, false_positives]):
                os.makedirs(f"{folder_path}/{name}/zeroshot/", exist_ok=True)
                data.to_csv(f"{folder_path}/{name}/zeroshot/{filename}")
        false_positive_ids.update(false_positives.index)
        false_negative_ids.update(false_negatives.index)
        if overlapping_false_positive_ids is None:
            overlapping_false_positive_ids = set(false_positives.index)
        else:
            overlapping_false_positive_ids.intersection_update(false_positives.index)
        if overlapping_false_negative_ids is None:
            overlapping_false_negative_ids = set(false_negatives.index)
        else:
            overlapping_false_negative_ids.intersection_update(false_negatives.index)
    print("#" * 50)
    print_padded_text("Total") 
    print_padded_text(f"False positives: {len(false_positive_ids)}")
    print_padded_text(f"False negatives: {len(false_negative_ids)}")
    print_padded_text(f"Overlapping false positives: {len(overlapping_false_positive_ids)}")
    print_padded_text(f"Overlapping false negatives: {len(overlapping_false_negative_ids)}")
    print("#" * 50)
    for filename, ids in zip(filenames, [overlapping_false_negative_ids, overlapping_false_positive_ids]):
        os.makedirs(f"{folder_path}/errors/", exist_ok=True)
        filtered_data = [
            result[result.index.isin(ids)]
            for result in results
        ]
        filtered_data = [
            data.rename(columns={"score": f"{model_name}_score", "reasoning": f"{model_name}_reasoning"})
            for data, model_name in zip(filtered_data, model_names)
        ]
        all_data = pd.concat(filtered_data, axis=1)
        columns = [
            label_column_name, 
            text_column_name, 
            *[f"{name}_score" for name in model_names], 
            *[f"{name}_reasoning" for name in model_names]
        ]
        all_data = all_data.loc[:, ~all_data.columns.duplicated()][columns]
        all_data.to_csv(f"../results/{folder_path}/errors/{filename}")