"""This module contains utils related to the plotting of the results, as
well as exploration of the data. It uses both matplotlib, seaborn and 
wordcloud to create plots/figures."""

from typing import List
from wordcloud import WordCloud
import pandas as pd
import os
import matplotlib.pyplot as plt
import stopwords

def dataframe_to_text(data: pd.DataFrame, labels: List[str]):
    """Extract text from a dataframe which can be used to create a word cloud"""
    text = ""
    for label in labels:
        text += data[label].str.cat(sep=" ")
    return text

def show_word_cloud(text: str, title: str, file_path = None):
    """Shows a word cloud based on the provided dataframe and labels
    
    Parameters
    ---------------------------
    text: str
        The text to be used for the word cloud
    title: str
        The title of the plot
    file_path: str (default = None)
        The file path to save the plot
    """
    exists = os.path.exists(file_path)
    if exists:
        word_cloud = plt.imread(file_path)
    word_cloud = WordCloud(stopwords=stopwords.get_stopwords('en')).generate(text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")
    if not exists and file_path is not None:
        plt.imsave(file_path, word_cloud.to_array())
    plt.show()