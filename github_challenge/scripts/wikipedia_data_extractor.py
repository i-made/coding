'''
Author : Nikhil Kulkarni
Pupose : Gihub Data Science Interview
Date   : February 11, 2017
Desc   : Extracts wikipedia data given entities
Data   : Requires keywords.csv file which has entities
Run    : python scripts/wikipedia_data_extractor.py data_files/keywords.csv
Read   : Generates a json file in 'data_files' folder with wiki text
'''

import re
import sys
import json
import pandas as pd

# Wiki api initialization
from wikiapi import WikiApi
wiki = WikiApi({'locale': 'en'})


def get_entities_from_csv(entity_filename="data_files/keywords.csv"):
    # reads keywords.csv file
    file_data = pd.read_csv(entity_filename)
    # words variable stores entities in a list
    words = list(set(file_data[file_data.columns[0]]))
    return words


def extract_wiki_data(words):
    # iterates over entities and fetches wikipedia data
    res = dict()
    for i in words:
        print 'Fetching data for: ', i
        results = wiki.find(i)
        if results:
            # Api returns multiple matches, this part combines them into one
            text = ''
            for j, q in enumerate(results):
                article = wiki.get_article(results[j])
                t = article.content.encode('ascii', 'ignore')
                t = " ".join(t.split())
                text += ' ' + t
            # Combined text of all possible terms is assigned
            text = text.replace(',', ' ')
            text = text.strip()
            res[i] = text
    return res


def json_file_writer(res, filename="data_files/wikidata.json"):
    # Dumps data into json/csv file
    data = pd.DataFrame(res.items(), columns=['Entity', 'Text'])
    # data.to_csv('WikiData.csv')
    with open(filename, "w") as file:
        file.write(json.dumps(res, file, indent=4))


def main():
    csv_file_name = sys.argv[1]
    words = get_entities_from_csv(csv_filename)
    # res has wikipedia corpus in a dictionary
    res = extract_wiki_data(words)
    # writes the data to a json file in data_files folder
    json_file_writer(res)
    return res

if __name__ == '__main__':
    main()
