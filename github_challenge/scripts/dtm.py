'''
Author : Nikhil Kulkarni
Pupose : Gihub Data Science Interview
Date   : February 11, 2017
Desc   : Generates Document Term Matrix (DTM)
Data   : Requires wikipedia data JSON file 'wikidata.json'
Run    : python scripts/dtm.py data_files/keywords.csv data_files/wikidata.json
Read   : Generated DTM in 'data_files' folder has [Documents x Terms] with (tf * idf) freq
'''

import re
import sys
import json
import numpy as np
import pandas as pd
from pandas import DataFrame

from wikipedia_data_extractor import get_entities_from_csv
from sklearn.feature_extraction.text import TfidfTransformer


def clean_dtm(df):
    "Cleans TDM by removing columns with all zeros"
    ddf = df.astype(float, errors='ignore')
    ddf = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
    return ddf


def get_tfidf_matrix(data_matrix):
    "Returns TFIDF matrix"
    tfidf = TfidfTransformer()
    data = tfidf.fit_transform(data_matrix)
    df = pd.DataFrame(data.todense())
    return df


class DTM():

    # This class handles generation of Document term matrix
    # It requires a dictionary where key is a title and value is the text of document
    # i.e. data = { "csharp": "C-Sharp is a programming ....", "nlp": "..."...}

    def __init__(self, corpus_dictionary, entities):
        # data = pd.read_csv('./Wiki_Data.csv')

        self.title_list = ['Doc:' + i for i in corpus_dictionary.keys()]
        self.abstract_list = corpus_dictionary.values()
        self.entity_list = entities

    def numpy_pandas_csvwriter(self):

        # Initiates empty matrix
        rows = len(self.abstract_list)
        columns = len(self.entity_list)
        term_matrix = np.zeros([columns, rows])

        print 'Constructing %d X %d Matrix' % (columns, rows)

        for abs_idx, abstract in enumerate(self.abstract_list):
            print abs_idx, len(self.abstract_list)
            for ent_idx, entity in enumerate(self.entity_list):
                entities = list(set([entity, entity.replace('-', ' ')]))
                number = 0
                for en in entities:
                    try:
                        number = number + len(
                            re.findall(
                                r'\b%s\b' %
                                en,
                                abstract,
                                re.IGNORECASE
                            )
                        )
                    except Exception as e:
                        print e, entity
                        number = number + 0
                term_matrix[ent_idx][abs_idx] = number

        # Generates pandas dataframe
        df = self.pandas_to_csv(term_matrix.transpose())
        return df

    def pandas_to_csv(self, matrix):

        rows = len(self.abstract_list)
        columns = len(self.entity_list)

        df = pd.DataFrame(
            matrix,
            columns=[entity for entity in self.entity_list],
            index=[entity for entity in self.title_list]
        )

        index_name = 'Entity'
        df.index.name = index_name
        #df.to_csv('DTM_%d_%d.csv' % (rows, columns), index=True)

        # Cleans Empty columns and get TF*IDF frequencies of entities
        df = clean_dtm(df)
        df = get_tfidf_matrix(df)

        # Outputs a Document Term matrix
        df.to_csv('DTM_%d_%d.csv' % (rows, columns), index=True)
        print '\n\nDone.'
        return df


def main():
    if sys.argv[1] and sys.argv[2]:
        entity_filename = sys.argv[1]
        corpus_filename = sys.argv[2]
    else:
        print 'First Arg: Entity CSV file'
        print 'Second Arg: Corpus JSON file'

    # imports variable words which is the list of input entities
    # i.e. words = ['c#','machine-learning','csharp',....]
    entities = get_entities_from_csv(entity_filename)

    # wikidata is the dictionary file with titles as keys and docs as values
    with open(corpus_filename) as json_data:
        corpus_dictionary = json.load(json_data)

    print len(entities)
    print len(corpus_dictionary.keys())

    # calls TDM Class
    dtm_extraction = DTM(corpus_dictionary, entities)
    dtm_extraction.numpy_pandas_csvwriter()


if __name__ == '__main__':
    main()
