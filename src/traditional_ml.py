"""This module implements training of traditional ML models, including
Random Forest and SVM classifiers. It is based on the implementation by Hassan 
et al. 2017 (https://dl.acm.org/doi/10.1145/3097983.3098131) 
What are used as features are the provided features in the original dataset: 
sentiment and length in addition to td-idf features, POS tags, and entity types."""

from typing import Iterable
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from claimbuster_utils import load_claimbuster_dataset
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler
import random
import numpy as np
import nltk
from nltk import pos_tag_sents, word_tokenize
import xgboost 

POS_TAGS = [
    "MD",
    "VBN",
    "PRP$",
    "CD",
    "NNS",
    "RBR",
    "LS",
    "WP",
    "JJR",
    "RB",
    "WP$",
    "VBZ",
    "-LRB-",
    "CC",
    "JJ",
    "$",
    ":",
    "VBG",
    "''",
    ",",
    "WDT",
    "EX",
    "PDT",
    "RP",
    "``",
    "NNPS",
    "NNP",
    "FW",
    "VB",
    "PRP",
    "RBS",
    "DT",
    "WRB",
    "NN",
    ".",
    "-NONE-",
    "IN",
    "TO",
    "UH",
    "VBD",
    "POS",
    "VBP",
    "JJS",
    "SYM",
    "(",
    ")",
]

class DropTextFeature(BaseEstimator, TransformerMixin):
    """Drops the text feature"""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        return x.drop("Text", axis=1)
    
class SentenceLengthFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sentence length feature extractor"""

    def fit(self, x: pd.DataFrame, y=None):
        return self
    
    def transform(self, x: pd.DataFrame):
        return x.assign(sentence_length = x["Text"].str.split().str.len())

class POSFeatureExtractor(BaseEstimator, TransformerMixin):
    """POS feature extractor"""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x = pd.DataFrame(x)

        pos_values = [
            [pos_tag for _, pos_tag in pos_tags]
            for pos_tags in pos_tag_sents(
                x["Text"].map(lambda x: nltk.word_tokenize(x))
            )
        ]
        pos_counts = pd.DataFrame(columns=POS_TAGS, index=x.index)
        for i, pos_list in enumerate(pos_values):
            pos_counts.iloc[i] = [
                pos_list.count(pos) for pos in POS_TAGS
            ]
        return pd.concat([x, pos_counts], axis=1)


class TfidFeatureExtractor(BaseEstimator, TransformerMixin):
    """Tfid feature extractor"""

    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, x: pd.DataFrame, y=None):
        x = pd.DataFrame(x)
        self.vectorizer.fit(x["Text"])
        return self

    def transform(self, x: pd.DataFrame):
        x = pd.DataFrame(x)
        tf_id_values = self.vectorizer.transform(x["Text"]).toarray()
        tf_id_names = self.vectorizer.get_feature_names_out()
        tf_id_data = pd.DataFrame(tf_id_values, columns=tf_id_names, index=x.index)
        data = pd.concat([x, tf_id_data], axis=1)
        return data


class ClaimbusterPipeline(Pipeline):
    """Claimbuster pipeline"""

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def fit(self, x: pd.DataFrame(), y: pd.Series):
        super().fit(x, y)

    def predict(self, x: pd.DataFrame()):
        return super().predict(x)


class DebugClassifier(BaseEstimator, TransformerMixin):
    """Used to inspect the pipeline"""

    def fit(self, x: pd.DataFrame(), y: pd.Series):
        return self

    def transform(self, x: pd.DataFrame()):
        print(f"{x.head()=}")
        return x


class RandomClassifier:
    """A classifier that predicts according to the distribution of the data"""

    def __init__(self) -> None:
        self.target_distribution = None

    def fit(self, _: pd.DataFrame(), y: pd.Series):
        """Fits the random classifier according to the target distribution"""
        self.target_distribution = y.value_counts()

    def predict(self, _: pd.DataFrame()):
        """Predicts the target distribution"""
        return random.choices(
            list(self.target_distribution.keys()),
            list(self.target_distribution.values()),
        )


def main():
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")
    extractor = TfidFeatureExtractor(stop_words="english")
    scalar = MaxAbsScaler()
    drop_text_feature = DropTextFeature()

    pipeline = ClaimbusterPipeline(
        [
            ("pos", POSFeatureExtractor()),
            # ('debug1', DebugClassifier()),
            ("word length", SentenceLengthFeatureExtractor()),
            ("tfidf", extractor),
            ("drop text feature", drop_text_feature),
            ("scalar", scalar),
            ("predictor", xgboost.XGBClassifier()),
        ]
    )
    data = load_claimbuster_dataset("data/ClaimBuster_Datasets/datasets")[
        ["Text", "Verdict"]
    ]
    x, y = data[["Text"]], data["Verdict"]
    print(cross_validate(pipeline, x, y, cv=5, scoring=["f1_macro", "accuracy"]))


if __name__ == "__main__":
    main()
