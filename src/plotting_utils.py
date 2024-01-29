"""This module contains utils related to the plotting of the results, as
well as exploration of the data. It uses both matplotlib, seaborn and 
wordcloud to create plots/figures."""

from typing import List, Iterable
from matplotlib.patches import Patch
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


def show_word_cloud(text: str, title: str, file_path: str = None):
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
    else:
        word_cloud = WordCloud(stopwords=stopwords.get_stopwords("en")).generate(text)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    if not exists and file_path is not None:
        plt.imsave(file_path, word_cloud.to_array())
    plt.show()


def show_histogram_plot(
    x: Iterable,
    y: Iterable,
    xlabel: str = None,
    ylabel: str = None,
    file_path: str = None,
):
    """Shows a histogram plot

    Parameters
    ---------------------------
    x: list
        The x values
    y: list
        The y values
    xlabel: str (default = None)
        The x label
    ylabel: str (default = None)
        The y label
    file_path: str (default = None)
        The file path to save the plot
    """
    plt.bar(x, y)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()


def show_sub_plots_with_legends(
    values: Iterable[float | int],
    group_size: int,
    titles: Iterable[str],
    legends: Iterable[str],
    ylabel: str,
    colors: Iterable[str],
    file_path: str = None,
):
    """Shows sub plots with legends. It assumes that the y axis is shared and
    that the values are grouped in bacthes of the same length as tit"""
    indices = range(0, len(values))
    color_map = dict(zip(indices, colors))
    fig, axs = plt.subplots(
        nrows=1, ncols=len(titles), sharex=False, sharey=True
    )
    for ax, title in zip(axs, titles):
        ax.set(title=title)
    for i in range(0, len(values), group_size):
        ax = axs[i // group_size]
        x = indices[i : i + group_size]
        y = values[i : i + group_size]
        ax.bar(x, y, color=colors)
        ax.get_xaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel(ylabel)
    patches = [Patch(color=color, label=label) for label, color in color_map.items()]
    fig.legend(
        labels=legends,
        handles=patches,
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
        frameon=False,
    )
    plt.tight_layout()
    plt.show()
    if file_path is not None:
        fig.savefig(file_path, bbox_inches="tight")


def show_sub_plots_pie_chart(
    values: Iterable[float | int],
    group_size: float,
    titles: Iterable[str],
    labels: Iterable[str],
    file_path: str = None,
):
    """Shows sub plots with pie charts"""
    fig, axs = plt.subplots(
        nrows=1, ncols=len(values) // group_size, sharex=False, sharey=True
    )
    for ax, title in zip(axs, titles):
        ax.set(title=title)
    for i in range(0, len(values), group_size):
        ax = axs[i // group_size]
        x = values[i : i + group_size]
        ax.pie(x, labels=labels, autopct="%.1f%%")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
    if file_path is not None:
        fig.savefig(file_path)