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



> Sorting the Similarity column in 'results_verbose.csv' will reveal top matches which look promising.



### Reproducing Results

Inside parent folder:

##### Extract wikipedia data for corresponding entities using

````
python scripts/wikipedia_data_extractor.py data_files/keywords.csv
````

> This will create a wikidata.json file in the 'data_files' folder

##### Generate Document Term Matrix for this corpus dictionary using

````
python scripts/dtm.py data_files/keywords.csv data_files/wikidata.json
````

> This will create a DTM CSV file in 'data_files' folder

##### Use this DTM to generate final results using

````
python scripts/gcn.py data_files/DTM_110_114.csv > results.csv
````
> This will create the desired results.csv file in the same folder.

> Please change variable 'thresh' to get more matching the default value is 0.7
> By experimentation it was found that 0.37 gives less false positives while avoiding false negatives. Provided results are for thresh=0.37

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

I planned to represent each entity as a vector. This vector will be constructed based on a set of documents. Each value in the vector will be TF*IDF score of that entity in a particular document. For this, I needed a corpus of documents.

I have used [Wikipedia API](https://github.com/richardasaurus/wiki-api) for generating a corpus. Searching each given entity, resulted in matching Wikipedia results. For each result, the method appends its text to yield a combined text. See 'data_files/Wiki_Data.csv' Or 'data_files/wikidata.json'

Using this I generated Document Term Matrix and at this point, I tried two approaches
1. K-Means clustering
2. Cosine Similarity

***K-Means Clustering*** approach didn't show promising results. I used [Silhouette Score](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) for selecting the best number of clusters.

Following code block was used for K-means clustering and analysis.


~~~~
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# df is the pandas data frame representing matrix
# k is the number of clusters

def get_clusters(df, k):
    "Returns clusters against each document"
    mat = df.as_matrix()
    km = KMeans(n_clusters=k)
    km.fit(mat)
    labels = km.labels_
    silhouette_avg = silhouette_score(mat, labels)
    results = pd.DataFrame([df.index, labels]).T
    return results, silhouette_avg
~~~~
~~~~
# iterates over different K value
for i in range(10, 40):
    results, score = get_clusters(clean_tdm(df), i)
    print i, score
~~~~

My own implementation of K-Means clustering is available [here](https://github.com/nikhilkul/info-extraction/blob/master/kmeans_info.py)

Since the formed clusters using this method did not show satisfactory results, I decided to compute [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between each vector to find which vectors are similar.

***Cosine Similarity*** was computed using the TF*IDF sparse matrix. It was converted to a dense representation using [LIL method](https://en.wikipedia.org/wiki/Sparse_matrix). Then the similarity was analyzed. 'gcn.py' gives the desired output for a threshold value of similarity. After plotting the histogram of different similarity values I found that at 0.37 the slope of the curve was high. All the results provided are for similarity > 0.37

I found 110 documents for 114 entities since four of the entities like 'data engineer' do not have a page on Wikipedia. Data Term Matrix has a shape f  110 X 114 (sparse representation)

*Answers to the following questions are given in the subsequent sections:*

>What are the pros and cons of your approach?

>Which terms do you get correct, which do you get incorrect, and why?

>What trade-offs (if any) did you make and why?


#### Pros
Using this approach, entities like 'precision', 'recall' can be matched! 

Even though these entities do not share any letter similarity, they are used in similar context. This context is captured by the vector representation of entities. 

Wikipedia is a credible source of data and the results using this method look promising even if the words are widely used in English. e.g. the word 'spark' is correct with other technical entities even though 'spark' also has a different meaning in English. Words like word2vec are unusual since it has a digit sandwiched between letters. 

Finding the good match for this entity would be difficult by a conventional method. But since we are converting word to a vector it becomes easy to represent the entities in terms of numbers and get comparison based on context. The context of 'embeddings' and 'word2vec' correctly matches here which is pleasantly surprising!

#### Cons

This method is highly dependent on the data and assumes that entities will be seen in the corpus. If the entity is absent in the corpus the similarity which is nothing but a dot product between two vectors will be zero. Here, for example, 'Hadoop me' is never seen in Wikipedia. This means the vector representing 'hadoop mr' is all zeros. 
~~~~
'hadoop mr' = [0 0 0 0 0 0 0 0 .... ]
~~~~
Dot product of a zero vector with any vector will be zero and the similarity will always be zero. Another thing to notice here is that the frequency of the word in a document plays a role. if all the documents mention one particular word too many times then that particular word will create false positives.

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
| c sharp                  | c#                               | 0.868981889 |
|graphical models          | probabilistic graphical models | 0.849254845 |
|ai                      | artificial intelligence           | 0.837217883 |
| word embeddings          | probabilistic models           | 0.817830435 |
|natural language processing| natural language               | 0.796664311 |
|f measure                  | precision                       | 0.77692418  |
|spark                      | mllib                           | 0.770578683 |
|spark                      | graphx                           | 0.770578683 |
|spark                      | apache spark                   | 0.768646245 |
|f measure                  | recall                           | 0.761517577 |
|linear regression          | regression                       | 0.734350344 |
|f score                  | recall                           | 0.728816146 |
|c sharp                  | csharp                           | 0.716975001 |
|precision                  | recall                           | 0.711147686 |
|f score                  | precision                       | 0.704043164 |    

This similarity table produces following results in the final set with threshold 0.37

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

Even though most of the words seem correct, one thing should be noticed that number of false negatives can be high. Words like python2, python3 etc. have no pages linked to them on Wikipedia. Since their vector representation is an all zero vector they won't be shown as similar to any entity. Words like 'mllib','graphx' appear together with 'spark','apache-spark' since they are under the same project 'apache-spark'. The desirability of such a match is debatable. Also note that 'Hadoop' appears with 'MapReduce' since they often appear together. Match of 'software' with 'application' is particularly interesting since they don't share a lot letters between them.

#### Trade-offs
Matching acronyms proved difficult especially with the presence of 'R'. 'R' generated false positives for abbreviations e.g. acronym of 'representation','recommendation' etc. is 'R'.
To deal with this issue, I have added a special rule. A similarity of 1 is assigned for abbreviations which might not be true always but works most of the time.
I have also replaced '-' in the term with space so that natural-language becomes natural language. This helped in searching entity in text and on wikipedia.

## **Performance**

#### Time Complexity

Time complexity in the generation of Document Term Matrix and similarity search remains O(n^2) but with more time it can be reduced further.

#### Space Complexity 

O(k) * 3 where k is the number of non-zero elements in the sparse representation of the dense document term matrix. 3 columns are needed to represent i,j indexes and non-zero value.

Sparse representation of a dense matrix in the LIL (list of lists) form reduces both time and space complexity of the search part. This approach has shown significant improvement in the running time for less number of entities and will certainly be helpful if there are a large number of entities.

### Future Scope

Answers to the following questions are given in the subsequent sections:

> What are your thoughts on future work for this project?

> What considerations would you need to make for scaling up the project?

> What trade-offs (if any) did you make and why?

#### Scaling Up
During my professional work experience, I have worked upon a similar problem statement where we were trying to find what is the probability of knowing 'java' given the person knows 'c++'. In that particular project, I looked at the relations between entities  e.g. 'is a' relation, 'has a' relation. Relations can help solve some hidden entity matching problem e.g. "'graphx' is under 'apache-spark' project" suggests that 'apache-spark' is the parent entity for 'graphx'. We can add features like these to the existing vectors and then do feature engineering using standard techniques to get the best precision for a known set of data. The addition of data from the different source will play an important role. We can assign confidence to each data source and depending upon the confidence we can then assign feature weights. e.g. features from Wikipedia get high weight.


#### To do

The Wikipedia data fetching can be done offline. Matrix creation can also be done in the dense format at its birth. Indexing the documents will help in the search process. Databases like elastic search are indexed using Lucene and can be used effectively to find frequencies. Additionally, spark computes word frequencies effeciently and can be clustered making it ideal to for scaling.

#### Key Element Optimization

For scaling up, I will optimize the matrix generation step since it is the slowest among all. Also, newer tools like spark allow matrix operations to be performed. Right now, for modularity, I have written the matrix in CSV and then loaded it again. This step can be avoided. Some documents have no words while some documents can convey the same information. Such documents can be identified using feature reduction techniques such as finding variance withing a feature, finding linear combination etc. This will help in reducing the size of the matrix and improving results.

### How to improve

Answers to the following questions are given in the subsequent sections:

>What would you do differently with more time? 

>What are the key elements that would require more thought? 

>What would you like to know or learn if you worked more on this project?

#### Active learning
Here, unlabeled data is abundant but manually labeling is expensive, learning algorithms can actively query the user/teacher for labels. This will enable us to use active learning techniques. Online learning can also produce interesting result since the data is constantly being updated. Given enough time I would create a graph database of all the entities and assign probabilities to each relation between nodes. Using online learning these weights will constantly get updated and produce similar entities through graph traversals. I would like to learn more about this type of learning.

#### Choosing knowledge base and performance metrics
Choice of data is essential and learned vector embeddings for each entity are heavily dependent upon the choice of data. This part certainly requires more thought. We will need to make sure that the data represents all of the entities and considers variations globally and not locally. The measurement of accuracy is also a key element that will need more thought since we do not have labels. Definition of performance matrices for models is crucial.

#### Links to datasets
I would like to know the algorithms and techniques used for solving such problems on the scale. With a size of the data, this problems becomes more computationally expensive to solve. One of the datasets I find useful for this particular use case is Wikipedia's 'redirects' dataset. This data is very rich and can be helpful in entity disambiguation.
Find few links to redirects data below: 
[c#](http://dispenser.homenet.org/~dispenser/cgi-bin/rdcheck.py?page=C-sharp),
[machine learning](http://dispenser.homenet.org/~dispenser/cgi-bin/rdcheck.py?page=machine_learning)
[mysql](http://dispenser.homenet.org/~dispenser/cgi-bin/rdcheck.py?page=mysql)
etc.

For MySql we can see that we get important matches from the redirect links itself.



