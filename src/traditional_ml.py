"""This module implements training of traditional ML models, including
Random Forest and SVM classifiers. It is based on the implementation by Hassan 
et al. 2017 (https://dl.acm.org/doi/10.1145/3097983.3098131) 
What are used as features are the provided features in the original dataset: 
sentiment and length in addition to td-idf features, POS tags, and entity types."""

import sklearn
from sklearn.svm import SVC
import pandas as pd
from claimbuster_utils import load_claimbuster_dataset
from sklearn.model_selection import cross_val_score, train_test_split

class RandomClassifier():
    """A classifier that predicts according to the distribution of the data"""

    def __init__(self) -> None:
        self.target_distribution = {}

    def fit(self, _: pd.DataFrame(), y: pd.Series):
        """Fits the random classifier according to the target distribution"""
        self.target_distribution = y.value_counts()

    def predict(self, _: pd.DataFrame()):
        """Predicts the target distribution"""
        return self.target_distribution
    



    

def main():
    svm = SVC()
    data = load_claimbuster_dataset("data/ClaimBuster_Datasets/datasets")["Text", "Length", "Sentiment", "Verdict"]
    x, y = data["Text"], data["Verdict"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    svm.fit(x_train, y_train)
    cross_val_score(svm, x, y, cv=5)

    pass

if __name__ == "__main__":
    main()