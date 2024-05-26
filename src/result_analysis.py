"""This module contains functions used for error analysis,
including exploration of false positives and false negatives"""

from typing import Iterable
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def flatten_classification_report(report: dict, drop_support_columns = True) -> dict:
    """Flattens a classification report generated from sklearn"""
    for key, value in report.copy().items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                report[f"{key}_{sub_key}"] = sub_value
            report.pop(key)
    if drop_support_columns:
        report = {k: v for k, v in report.items() if not k.endswith("support")} 
    return report

def print_padded_text(text: str, total_length = 50):
    """Pad with hastags at the beggining and end of the text and center it."""
    padding_length_left = (total_length-len(text))//2
    padding_length_right = total_length - len(text) - padding_length_left
    print(f"{'#': <{padding_length_left}}{text}{'#': >{padding_length_right}}")

def generate_error_analysis_report(
    data: pd.DataFrame,
    predictions: Iterable[pd.DataFrame],
    model_names: Iterable[str],
    folder_path: str = None,
    reasoning: pd.Series = None,
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
    for name, prediction in zip(model_names, predictions):
        prediction[label_column_name] = data.loc[prediction.index][label_column_name]
        prediction[text_column_name] = data.loc[prediction.index][text_column_name]
        false_negatives = prediction[prediction.apply(lambda x: x["prediction"]==0 and x[label_column_name]==1, axis=1)]
        false_positives = prediction[prediction.apply(lambda x: x["prediction"]==1 and x[label_column_name]==0, axis=1)]
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
    filenames = ["false_negatives.csv", "false_positives.csv"]
    for filename, ids in zip(filenames, [overlapping_false_negative_ids, overlapping_false_positive_ids]):
        os.makedirs(f"{folder_path}/errors/", exist_ok=True)
        filtered_data = [
            result[result.index.isin(ids)]
            for result in predictions
        ]
        filtered_data = [
            data.rename(columns={"prediction": f"{model_name}_prediction"})
            for data, model_name in zip(filtered_data, model_names)
        ]
        all_data = pd.concat(filtered_data, axis=1)
        if reasoning is not None:
            all_data["reasoning"] = reasoning.loc[all_data.index]
        columns = [
            label_column_name, 
            text_column_name, 
            *(["reasoning"] if reasoning is not None else []), 
            *[f"{name}_prediction" for name in model_names], 
        ]
        all_data = all_data.loc[:, ~all_data.columns.duplicated()][columns]
        all_data.to_csv(f"{folder_path}/errors/{filename}")

def create_confusion_matrix(
    original_data: pd.DataFrame,
    predictions: Iterable[float],
    label_column_name = "Verdict",
    save_path: str = None,
    show = False,
):
    """Display a confusion matrix"""
    y_true = original_data[label_column_name]
    conf_matrix = confusion_matrix(y_true, predictions)
    categories = ["True Neg", "False Pos", "False Neg", "True Pos"]
    values = conf_matrix.flatten()
    labels = [f"{category}\n{value}" for category, value in zip(categories, values)]
    labels = np.array(labels).reshape(2, 2)
    sns.heatmap(conf_matrix, annot=labels, fmt="000", )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()
