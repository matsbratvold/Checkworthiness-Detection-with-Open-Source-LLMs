""" In this module, code for cleaning datasets is provided. In particular, the WelFake dataset is cleaned."""

import pandas as pd
import numpy as np


def clean_welfake_dataset(save_to_csv=True):
    """Cleans the welfake dataset"""
    df: pd.DataFrame = pd.read_csv("data/WELFake/raw_dataset.csv", index_col=0)
    df = df.dropna()
    df = df.drop_duplicates()
    df["label"] = df["label"].replace({0: "REAL", 1: "FAKE"})
    df.replace(to_replace=r"[\n\t\r]", value="", regex=True, inplace=True)
    df = df.apply(
        lambda x: x.str.strip() if x.name in ["title", "text"] else x
    )
    # Drop rows with empty text or where text is not string
    df = df[df["text"].apply(lambda x: type(x) == str and len(x.strip()) > 0)]
    if save_to_csv:
        df.to_csv("data/WELFake/cleaned_dataset.csv")
    return df


def remove_outliers(
    df: pd.DataFrame,
    text_column: str = "text",
    text_length_min: int = 500,
    text_length_max: int = 10000,
    average_word_length_min: float = 4,
    average_word_length_max: float = 10,
    save_to_csv: bool = True,
):
    """Remove outliers from a dataframe based on the length of the text and the average word length
    Args:
    ---------
    df: pd.DataFrame
        The dataframe to clean
    text_column: str
        The name of the column containing the text to clean
    text_length_min: int
        The minimum length of the text
    text_length_max: int
        The maximum length of the text
    aver
    average_word_length_max: int
        The maximum average word length of the text
    """
    df = df[
        df[text_column].apply(
            lambda x: len(x) >= text_length_min and len(x) <= text_length_max
        )
    ]
    df = df[
        df[text_column].apply(
            lambda x: np.average([len(word) for word in x.split(" ")]) >= average_word_length_min
            and np.average([len(word) for word in x.split(" ")]) <= average_word_length_max
        )
    ]
    if (save_to_csv): 
        df.to_csv("data/WELFake/cleaned_no_outliers.csv")
    return df


def main():
    clean_welfake_dataset(True)


if __name__ == "__main__":
    main()
