## **Description**

### Overview


1. 'scripts' folder includes three scripts. These scripts are separated intentionally to make the code modular. 
2. 'data_files' folder includes all the data files that get created in the process
3. 'ouput_files' folder has the desired output along with verbose output in the following format



| Entity 1 |           Entity 2          | Similarity |
|:--------:|:---------------------------:|:----------:|
| nlp      | natural-language-processing |    0.996   |
| scipy    | numpy                       |    0.866   |
| lda      | latent-dirichlet-allocation |     1.0    |
| ...      | ...                         |     ...    |



> Sorting the Similarity column in 'output_verbose.csv' will reveal top matches which look promising.



### Reproducing Results

Inside parent folder:

##### Extract wikipedia data for corresponding entities using

`python scripts/wikipedia_data_extractor.py data_files/keywords.csv`

> This will create a wikidata.json file in the 'data_files' folder

##### Generate Document Term Matrix for this corpus dictionary using

`python scripts/dtm.py data_files/keywords.csv data_files/wikidata.json`

> This will create a DTM CSV file in 'data_files' folder

##### Use this DTM matrix to generate final results using

``

## **Guidelines**

  - [Description](#description)
    - [Overview](#overview)
    - [Reproducing results](#reproducing-results)
  - [Guidelines](#guidelines)
      - [Approach](#approach)
        - [Pros](#pros)
        - [Cons](#cons)
        - [Results and Discussion](#results-and-discussion)
      - [Performance](#performance)
        - [Time Complexity](#time-complexity)
        - [Space Complexity](#space-complexity)
      - [Future Scope](#future-scope)
        - [Scaling Up](#scaling-up)
        - [To do](#to-do)
        - [Key Element Optimisation](#key-element-optimisation)
      - [How to improve](#how-to-improve)
        - [Active learning](#active-learning)
        - [Choosing knowledge base](#choosing-knowledge-base)
        - [Links to datasets](#links-to-datasets)

## **Approach**

I am using [Wikipedia API](https://github.com/richardasaurus/wiki-api) for generating corpus

Answers to the following questions are given in the subsequent sections:

What are the pros and cons of your approach?

Which terms do you get correct, which do you get incorrect, and why?

What trade-offs (if any) did you make and why?

'R' generated false positives for abbreviations
Similarity of 1 is assigned for abbreviations


#### Pros
#### Cons
#### Results and Discussion
I am listing down some of the encouraging results:

| Entity 1               | Entity 2                       | Similarity  |
|------------------------|--------------------------------|-------------|
| data science           | data scientist                 | 0.996464625 |
| mllib                  | apache spark                   | 0.995573495 |
| apache spark           | graphx                         | 0.995573495 |
| f measure              | f score                        | 0.990980439 |
| recommendation system  | recommendation systems         | 0.98802305  |
| map reduce             | mapr                           | 0.986779179 |
| csharp                 | c#                             | 0.967993413 |
| graphical model        | probabilistic graphical models | 0.966150009 |
| graphical models       | graphical model                | 0.948068175 |
| hadoop cluster         | hadoop                         | 0.936946969 |
| recommendation system  | recommendations                | 0.931221224 |
| recommendation systems | recommendations                | 0.920268623 |
| c sharp                | c#                             | 0.868981889 |
|graphical models        | probabilistic graphical models | 0.849254845 |
|ai                    | artificial intelligence        | 0.837217883 |
| word embeddings        | probabilistic models         | 0.817830435 |
|natural language processing| natural language            | 0.796664311 |
|f measure               | precision                    | 0.77692418  |
|spark                   | mllib                        | 0.770578683 |
|spark                   | graphx                         | 0.770578683 |
|spark                   | apache spark                 | 0.768646245 |
|f measure               | recall                         | 0.761517577 |
|linear regression       | regression                     | 0.734350344 |
|f score               | recall                         | 0.728816146 |
|c sharp               | csharp                         | 0.716975001 |
|precision               | recall                         | 0.711147686 |
|f score               | precision                    | 0.704043164 |    

> This similarity table produces following results in the final set with threshold 0.3

| load                         | extract                        | transform                      | etl                          |                              |
|------------------------------|--------------------------------|--------------------------------|------------------------------|------------------------------|
| nlp                          | natural language processing    | natural language process       | natural language             |                              |
| lda                          | latent dirichlet allocation    |                                |                              |                              |
| unsupervised learning        | machine learning               |                                |                              |                              |
| ai                           | artificial intelligence        |                                |                              |                              |
| data science                 | data scientist                 |                                |                              |                              |
| convolutional neural network | cnn                            | neural network                 | neural net                   |                              |
| graphical model              | probabilistic graphical models |                                |                              |                              |
| clustering                   | cluster                        |                                |                              |                              |
| recommendation systems       | recommendations                | recommendation system          |                              |                              |
| pgm                          | probabilistic graphical models |                                |                              |                              |
| graph                        | graphical model                | probabilistic graphical models | graphical models             |                              |
| mllib                        | graphx                         | apache spark                   |                              |                              |
| c sharp                      | c#                             | csharp                         |                              |                              |
| hadoop cluster               | hadoop                         |                                |                              |                              |
| pca                          | principal component analysis   |                                |                              |                              |
| word embeddings              | probabilistic models           | embeddings                     | word2vec                     | principal component analysis |
| graphical models             | graphical model                | probabilistic graphical models | modeling                     |                              |
| embeddings                   | probabilistic models           | word2vec                       | principal component analysis |                              |
| scientific                   | scientific computing           |                                |                              |                              |
| f measure                    | f score                        | precision                      | recall                       |                              |
| cosine distance              | distance                       |                                |                              |                              |
| scipy                        | numpy                          |                                |                              |                              |
| spark                        | mllib                          | graphx                         | apache spark                 |                              |
| ml                           | machine learning               |                                |                              |                              |
| mapreduce                    | hadoop                         | mapr                           | map reduce                   | mr                           |
| software                     | application                    |                                |                              |                              |

'hadoop mr' doesn't appear in data

## **Performance**

#### Time Complexity

Time complexity in the generation of Document Term Matrix and similarity search remains O(n^2) but can be reduced further.

#### Space Complexity 

O(k) * 3 where k is the number of non zero elements in the sparse representation of the dense document term matrix. 3 columns are needed to represent i,j indexes and non zero value.

Sparse representation of dense matrix in the LIL (list of lists) form reduces both time and space complexity of the search part. This approach has shown significant improvement in the running time for less number of entities and will certainly be helpful if there are large number of entities.

### Future Scope

Answers to the following questions are given in the subsequent sections:

What are your thoughts on future work for this project?

what considerations would you need to make for scaling up the project?

What trade-offs (if any) did you make and why?

#### Scaling Up
#### To do
#### Key Element Optimisation

### How to improve

Answers to the following questions are given in the subsequent sections:

What would you do differently with more time? 

What are the key elements that would require more thought? 

What would you like to know or learn if you worked more on this project?

#### Active learning
#### Choosing knowledge base
#### Links to datasets





