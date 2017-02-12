'''
Author : Nikhil Kulkarni
Pupose : Gihub Data Science Interview
Date   : February 11, 2017
Desc   : Entity Matching
'''
import sys
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def main():
    CSV_PATH = sys.argv[1]
    df = pd.read_csv(CSV_PATH)
    df_index = df.set_index('Entity')
    df_transpose = df_index.transpose()

    # Entities
    entities = df_transpose.index.values.tolist()

    # Dataframe to work on
    numpyMatrix = df_transpose.as_matrix()
    A_sparse = sparse.csr_matrix(numpyMatrix)
    similarities = cosine_similarity(A_sparse)
    print len(similarities)
    print('pairwise dense output:\n {}\n'.format(similarities))

if __name__ == "__main__":
    main()
