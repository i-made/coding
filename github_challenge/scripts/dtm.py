'''
Author : Nikhil Kulkarni
Pupose : Gihub Data Science Interview
Date   : February 11, 2017
Desc   : Generates Document Term Matrix (DTM)
Data   : Requires wikipedia data JSON file
Run    : python dtm.py wikidata.json
Read   : Generated DTM has [Documents x Terms] with (tf * idf) freq
'''

import re
import sys
import json
import numpy as np
import pandas as pd
from pandas import DataFrame

# imports variable words which is the list of input entities
# i.e. words = ['c#','machine-learning','csharp',....]
from entities import words


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
    # It requires a dictionary where key is a term and value is the document
    # i.e. data = { "csharp": "C-Sharp is a programming ....", "nlp": "..."...}

    def __init__(self):
        # data = pd.read_csv('./Wiki_Data.csv')

        # wikidata is the dictionary file with terms as keys and docs as values
        with open('wikidata.json') as json_data:
            data = json.load(json_data)

        self.title_list = ['Doc:' + i.split()[0] for i in data.values()]
        self.abstract_list = data.values()
        self.entity_list = [i.replace('-', ' ') for i in words]

        # print len(self.entity_list)

    def numpy_pandas_csvwriter(self):

        # Initiates empty matrix
        rows = len(self.abstract_list)
        columns = len(self.entity_list)
        term_matrix = np.zeros([columns, rows])

        print 'Constructing %d X %d Matrix' % (columns, rows)

        for abs_idx, abstract in enumerate(self.abstract_list):
            print abs_idx, len(self.abstract_list)
            for ent_idx, entity in enumerate(self.entity_list):
                try:
                    number = len(re.findall(
                        r'\b%s\b' %
                        entity,
                        abstract,
                        re.IGNORECASE
                    ))
                except Exception as e:
                    number = 0.0
                term_matrix[ent_idx][abs_idx] = number

        # Generaes pandas dataframe
        df = self.pandas_to_csv(term_matrix.transpose())
        return df

    def pandas_to_csv(self, matrix):
        index_name = 'Entity'

        rows = len(self.abstract_list)
        columns = len(self.entity_list)

        df = pd.DataFrame(
            matrix,
            columns=[entity for entity in self.entity_list],
            index=[entity for entity in self.title_list]
        )
        df.index.name = index_name

        # title_df = pd.DataFrame(df['Entity'])
        # df = df.set_index('Entity')
        # df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        # df = clean_tdm(df)
        # df = get_tfidf_matrix(df)
        # df = pd.concat([title_df, df], axis=1)

        # Outputs a Document Term matrix
        df.to_csv('DTM_%d_%d.csv' % (rows, columns), index=True)
        print '\n\nDone.'
        return df


if __name__ == '__main__':
    tdm_extraction = TDM()
    tdm_extraction.numpy_pandas_csvwriter()
