""" In this module, New York Times Article Search API is used to get the articles.
This could be helpful to compare articles for fake news detection with articles from New York Times."""

import requests
import os
import json

API_KEY = os.environ.get('NEW_YORK_TIMES_API_KEY')
API_URL = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'

def search_for_articles(query, begin_date, end_date):
    """Search for articles using the New York Times Article Search API."""
    url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    params = {'api-key': API_KEY, 'q': query, 'begin_date': begin_date, 'end_date': end_date}
    response = requests.get(url, params=params)
    return response.json()

def search_in_archive(month: int, year: int):
    """Search for articles in the archive using the New York Times Article Search API."""
    url = f'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json'
    params = {'api-key': API_KEY}
    response = requests.get(url, params=params)
    return response.json()

def main(): 
    json_string: dict[str, list | dict] = search_for_articles("Republican tax cuts", "20171226", "20180103")
    # Drop multi media from json-string
    for doc in json_string["response"]["docs"]:
        doc.pop("multimedia", None)
    json.dump(json_string, open("new_york_times_mark-meadows.json", 'w'))
    # json_string: dict[str, list | dict] = search_in_archive(12, 2017)
    # json.dump(json_string, open("new_york_times_archive_2017_12.json", 'w'))

if __name__ == '__main__':
    main()