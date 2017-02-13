'''
Author : Nikhil Kulkarni
Pupose : Gihub Data Science Interview
Date   : February 11, 2017
Desc   : Performs Entity Matching and outputs clusters
Data   : Requires Term-Document-Matrix CSV file
Run    : python gcn.py TDM_111_115.csv > output.csv
Read   : Similarities are in descending order in output.csv
'''

import sys
import numpy as np
import pandas as pd
from scipy import sparse
from pprint import pprint
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

# Thresh is the cosine simililarity threshhold
thresh = 0.0
# Output has the need list of similar entities
output = list()
# Canonical unique entities
canonical = list()


def csv_output(sim_tuples):
        # This function will generate desired csv file
        # while using this function use '> output.csv'
        # i.e. python gcn.py TDM_111_115.csv > output.csv

    # Rearrange list of list in descending order of similarities
    for k, v in sim_tuples.items():
        sim_tuples[k] = sorted(v, key=itemgetter(1), reverse=True)

    # Uncomment following statement to see detained output dictionary
    # pprint(sim_tuples)

    for k, v in sim_tuples.items():
        z = ",".join([i[0] for i in v])
        print k, ',', z


def process_similarities(similarities, entities):
        # Dictionary which holds tuples
    siml = dict()
    # Canonical variable maintains list of canonical entities
    # This will help in deduplication of tuples
    canonical = list()

    # Iterates over similarity matrix and generates similarity tuples
    for i in range(0, len(similarities)):

            # Abbreviations are generated and checked
            # abv1 is the abbreviation of entitiy 1
        abv1 = "".join(k[0] for k in entities[i].split())

        for j, l in enumerate(similarities[i]):
            if l > thresh and entities[i] != entities[j]:

                    # abv2 is the abbreviation of entitiy 1
                abv2 = "".join(m[0] for m in entities[j].split())

                # 'R' generated false positives for abbreviations
                # Similarity of 1 is assigned for abbreviations
                if abv1 == entities[j] and abv1 != 'r':
                    l = 1.0
                if abv2 == entities[i] and abv2 != 'r':
                    l = 1.0
                if entities[j] not in canonical:
                    if entities[i] in siml.keys():
                        if entities[j] not in siml.keys():
                            siml[entities[i]].append([entities[j], l])
                    else:
                        siml[entities[i]] = [[entities[j], l]]

                # Uncomment following print statement to get output_verbose.csv
                # This csv will have all the entities and their similarities
                # Comment call of csv_output function
                # Run : python gcn.py TDM_111_115.csv > output_verbose.csv

                print entities[i], ",", entities[j], ",", l

        # After an entity is done it needs to be put in canonical for
        # depuplication
        canonical.append(entities[i])

    return siml


def get_similarities(df_transpose):
    # Generates dataframe to work on

    # CSV is converted to matrix where columns are docs and rows are entities
    numpyMatrix = df_transpose.as_matrix()
    # Since the CSV is sparse we convert it to its sparse representation
    A_sparse = sparse.csr_matrix(numpyMatrix)
    # We get cosine similarity matrix by taking dot product between vectors
    similarities = cosine_similarity(A_sparse)
    return similarities


def main():
    # This part reads the TFIDF matrix and takes its transpose
    CSV_PATH = sys.argv[1]
    df = pd.read_csv(CSV_PATH)
    df_index = df.set_index('Entity')
    df_transpose = df_index.transpose()

    # List of Entities
    entities = df_transpose.index.values.tolist()

    similarities = get_similarities(df_transpose)
    sim_tuples = process_similarities(similarities, entities)
    # csv_output(sim_tuples)


if __name__ == "__main__":
    main()
