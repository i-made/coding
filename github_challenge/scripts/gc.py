from tdm import *
from word_heatmap import *


def plot_main(cooccurrence_matrix, pl):
    print cooccurrence_matrix
    print pl
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(
            cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    # print(
    #     '\ncooccurrence_matrix_percentage:\n{0}'.format(
    #         cooccurrence_matrix_percentage)
    # )

    # Add count in labels
    label_header_with_count = [
        '{0} ({1})'.format(
            pl,
            cooccurrence_matrix_diagonal[label_number]
        ) for label_number, label_header in enumerate(pl)
    ]
    # print('\nlabel_header_with_count: {0}'.format(label_header_with_count))

    # Plotting
    x_axis_size = cooccurrence_matrix_percentage.shape[0]
    y_axis_size = cooccurrence_matrix_percentage.shape[1]
    title = "Co-occurrence matrix\n"
    xlabel = ''  # "Labels"
    ylabel = ''  # "Labels"
    xticklabels = label_header_with_count
    yticklabels = label_header_with_count
    heatmap(
        cooccurrence_matrix_percentage,
        title, xlabel, ylabel, xticklabels, yticklabels
    )
    # use format='svg' or 'pdf' for vectorial pictures
    plt.savefig('image_output.png', dpi=300, format='png', bbox_inches='tight')
    # plt.show()


def sparse_cosine(df):
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse
    pl = [i for i in df.columns.values if i != 'Entity']
    A_sparse = sparse.csr_matrix(df)
    similarities = cosine_similarity(A_sparse)
    print('pairwise dense output:\n {}\n'.format(similarities))

    # Also can output sparse matrices
    similarities_sparse = cosine_similarity(A_sparse)
    print similarities_sparse
    # print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


def main():
    CSV_PATH = sys.argv[1]
    df = pd.read_csv(CSV_PATH)
    df = df.transpose()
    sparse_cosine(df)
    # df.to_csv('final.csv')
    # title_df = pd.DataFrame(df['Entity'])
    # df = df.set_index('Entity')
    # df = clean_tdm(df)
    # df = get_tfidf_matrix(df)
    # df = pd.concat([title_df, df], axis=1)

    # df = df.transpose()
    # # df = get_coocc(df)

    # plot_main(df, pl)

if __name__ == "__main__":
    main()
