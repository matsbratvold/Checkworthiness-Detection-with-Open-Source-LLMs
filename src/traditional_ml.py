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
from checkthat_utils import load_check_that_dataset
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, make_scorer, accuracy_score
import random
import numpy as np
import nltk
from nltk import pos_tag_sents
from scipy.stats import logistic, randint
from result_analysis import flatten_classification_report


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

class RandomForestFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects the best features"""

    def __init__(self, k=100):
        self.k = k
        self.classifier = RandomForestClassifier(n_estimators=100) 

    def fit(self, x: pd.DataFrame, y=None):
        self.classifier.fit(x, y)
        return self

    def transform(self, x: Iterable):
        x = pd.DataFrame(x)
        return x.iloc[:, self.classifier.feature_importances_.argsort()[-self.k:]]

class DropTextFeature(BaseEstimator, TransformerMixin):
    """Drops the text feature"""

    def __init__(self, text_column="Text") -> None:
        self.text_column = text_column

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        return x.drop(self.text_column, axis=1)
    
class SentenceLengthFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sentence length feature extractor"""

    def __init__(self, text_column="Text") -> None:
        self.text_column = text_column

    def fit(self, x: pd.DataFrame, y=None):
        return self
    
    def transform(self, x: pd.DataFrame):
        return x.assign(sentence_length = x[self.text_column].str.split().str.len())

class POSFeatureExtractor(BaseEstimator, TransformerMixin):
    """POS feature extractor"""

    def __init__(self, text_column="Text") -> None:
        self.text_column = text_column

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        x = pd.DataFrame(x)

        pos_values = [
            [pos_tag for _, pos_tag in pos_tags]
            for pos_tags in pos_tag_sents(
                x[self.text_column].map(lambda x: nltk.word_tokenize(x))
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

    def __init__(self, text_column="Text", **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.text_column = text_column

    def fit(self, x: pd.DataFrame, y=None):
        x = pd.DataFrame(x)
        self.vectorizer.fit(x[self.text_column])
        return self

    def transform(self, x: pd.DataFrame):
        x = pd.DataFrame(x)
        tf_id_values = self.vectorizer.transform(x[self.text_column]).toarray()
        tf_id_names = self.vectorizer.get_feature_names_out()
        tf_id_data = pd.DataFrame(tf_id_values, columns=tf_id_names, index=x.index)
        data = pd.concat([x, tf_id_data], axis=1)
        return data


class ClaimbusterPipeline(Pipeline):
    """Claimbuster pipeline"""

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        super().fit(x, y)

    def predict(self, x: pd.DataFrame):
        return super().predict(x)


class DebugClassifier(BaseEstimator, TransformerMixin):
    """Used to inspect the pipeline"""

    def fit(self, x: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, x: pd.DataFrame):
        print(f"{x.head()=}")
        return x


class RandomClassifier:
    """A classifier that predicts according to the distribution of the data"""

    def __init__(self) -> None:
        self.target_distribution = None

    def fit(self, _: pd.DataFrame, y: pd.Series):
        """Fits the random classifier according to the target distribution"""
        self.target_distribution = y.value_counts()

    def predict(self, _: pd.DataFrame):
        """Predicts the target distribution"""
        return random.choices(
            list(self.target_distribution.keys()),
            list(self.target_distribution.values()),
        )

def classification_report_scorer(clf, X, y):

    y_pred = clf.predict(X)
    # Generate columns by flattening the classification_report
    return flatten_classification_report(
        classification_report(y, y_pred, output_dict=True)
    )
def main():
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")
    text_column = "tweet_text"
    label_column = "check_worthiness"
    tfidf_extractor = TfidFeatureExtractor(stop_words="english", text_column=text_column)
    scalar = MaxAbsScaler()
    pos_extractor = POSFeatureExtractor(text_column)
    drop_text_feature = DropTextFeature(text_column)
    classifier = svm.SVC()

    pipeline = ClaimbusterPipeline(
        [
            # ('debug1', DebugClassifier()),
            ("sentence length", SentenceLengthFeatureExtractor(text_column)),
            ("tfidf", tfidf_extractor),
            ("drop text feature", drop_text_feature),
            ("scalar", scalar),
            ("predictor", classifier),
        ]
    )
    data = load_check_that_dataset("data/CheckThat")[
        [text_column, label_column]
    ]
    data = pos_extractor.transform(data)
    print(data.head())
    x, y = data.drop(label_column, axis=1), data[label_column]
    param_distributions = {
        "predictor__C": logistic(1, 0.5),
    }
    search_cv = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        random_state=0,
        n_iter=10
    )
    best_params = search_cv.fit(x, y).best_params_
    print(best_params)
    pipeline.set_params(**best_params)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    result = pd.DataFrame(
        cross_validate(pipeline, x, y, cv=splitter, scoring=classification_report_scorer)
    )
    result.loc["Average"] = result.mean()
    print(result)


if __name__ == "__main__":
    main()
