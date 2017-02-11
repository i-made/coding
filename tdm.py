# -*- coding: UTF-8 -*-
#!/usr/bin/env python
'''
Author: Nikhil Kulkarni (nikhil.kulkarni@innoplexus.com)
Usage: python tdm.py
'''
import re
import csv
import time
import string
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame

words = [
    "similarity",
    "gensim",
    "data-science",
    "math",
    "prediction",
    "hadoop-platform",
    "representation",
    "analytical",
    "neural-network",
    "big-data",
    "natural-language-processing",
    "recommendation-systems",
    "evaluation",
    "spark-stream",
    "natural-language",
    "cluster",
    "scientific-computing",
    "tensorflow",
    "ai",
    "scipy",
    "pyspark",
    "scientist",
    "data-engineer",
    "dimention",
    "natural-language-algorithm",
    "precision",
    "map-reduce",
    "numpy",
    "linear-regression",
    "graphical-model",
    "spark-shell",
    "tensor",
    "supervised-learning",
    "probabilistic",
    "data-engineering",
    "spark",
    "cosine-distance",
    "c#",
    "pca",
    "csharp",
    "probabilistic-graphical-models",
    "modeling",
    "statistical",
    "regression",
    "apache-spark",
    "extract",
    "metrics",
    "python",
    "mapr",
    "mr",
    "fscore",
    "python2",
    "recall",
    "application",
    "developer",
    "networkx",
    "principal-component-analysis",
    "ica",
    "probabilistic-models",
    "c-sharp",
    "recommendation-system",
    "analytics",
    "scientific",
    "etl",
    "mapreduce",
    "lda",
    "engineer",
    "transform",
    "react",
    "distance",
    "logistic-regression",
    "graphx",
    "neural-net",
    "software",
    "load",
    "word2vec",
    "artificial-intelligence",
    "javascript",
    "deep-learning",
    "hadoop",
    "python3",
    "extract-load-transform",
    "information-retrieval",
    "graph",
    "graphical-models",
    "signal",
    "f-score",
    "matrix",
    "features",
    "apachespark",
    "f-measure",
    "dimension",
    "mllib",
    "unsupervised-learning",
    "r",
    "word-embeddings",
    "embeddings",
    "vector",
    "hadoop-cluster",
    "data-scientist",
    "recommendations",
    "clustering",
    "hadoop-mr",
    "data-processing",
    "ml",
    "machine-learning",
    "reinforcement-learning",
    "latent-dirichlet-allocation",
    "mrjob",
    "natural-language-process",
    "nlp",
    "f1score",
    "pgm",
    "cnn",
    "convolutional-neural-network"]

# -*- coding: UTF-8 -*-
#!/usr/bin/env python
'''
Author: Nikhil Kulkarni (nikhil.kulkarni@innoplexus.com)
Date: 17 March 2016
Usage: python clustering.py
'''
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_samples, silhouette_score


def clean_tdm(df):
    "Cleans TDM by removing first column and columns with all zeros"
    ddf = df.astype(float, errors='ignore')
    ddf = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
    # L, S, svd = pcp(ddf, verbose=True, svd_method="exact")
    # print L
    # print S
    ddf.to_csv('new_file.csv', index='pubmed_id')
    return ddf


def get_coocc(data_matrix):
    "Returns cooccurence matrix : X'X "
    return data_matrix.T.dot(data_matrix)


def get_tfidf_matrix(data_matrix):
    "Returns TFIDF matrix"
    tfidf = TfidfTransformer()
    data = tfidf.fit_transform(data_matrix)
    df = pd.DataFrame(data.todense())
    return df


def get_clusters(df, k):
    "Returns clusters against each document"
    mat = df.as_matrix()
    km = KMeans(n_clusters=k)
    km.fit(mat)
    labels = km.labels_
    silhouette_avg = silhouette_score(mat, labels)
    results = pd.DataFrame([df.index, labels]).T
    return results, silhouette_avg


class TDM():

    def __init__(self):

        data = pd.read_csv('./Wiki_Data.csv')
        self.title_list = ['Doc:' + i.split()[0] for i in data['Text']]
        self.abstract_list = data['Text']
        self.entity_list = [i.replace('-', ' ') for i in words]
        print len(self.entity_list)

    def numpy_pandas_csvwriter(self):
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
                        re.I
                    ))
                except Exception as e:
                    number = 0.0
                term_matrix[ent_idx][abs_idx] = number
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
        df.to_csv('TDM_%d_%d.csv' % (rows, columns), index=True)
        print '\n\nDone.'
        return df


if __name__ == '__main__':
    # tdm_extraction = TDM()
    # tdm_extraction.numpy_pandas_csvwriter()

    CSV_PATH = sys.argv[1]
    tp = pd.read_csv(CSV_PATH, iterator=True)
    df = pd.concat(tp, ignore_index=True)
    title_df = pd.DataFrame(df['Entity'])
    df = df.set_index('Entity')
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    print df.shape
    df = clean_tdm(df)
    df = get_tfidf_matrix(df)
    df = pd.concat([title_df, df], axis=1)
    df.to_csv('final.csv')
    for i in range(2, 20):
    results, score = get_clusters(clean_tdm(df), 4)
    print i, score
