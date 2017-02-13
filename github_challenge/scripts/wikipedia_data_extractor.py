import re
import json
import pandas as pd
from wikiapi import WikiApi
wiki = WikiApi()
wiki = WikiApi({'locale': 'en'})

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
    "convolutional-neural-network"
]

res = dict()
for i in words:
    print 'searched for ', i
    results = wiki.find(i)
    print results
    print '*' * 70
    if results:
        text = ''
        for j, q in enumerate(results):
            print results[j]
            article = wiki.get_article(results[j])
            t = article.content.encode('ascii', 'ignore')
            print t
            print '*' * 60
            t = " ".join(t.split())
            print '*' * 60
            text += ' ' + t
            print '-' * 60
            print text
        text = text.replace(',', ' ')
        text = text.strip()
        res[i] = text
data = pd.DataFrame(res.items(), columns=['Entity', 'Text'])
data.to_csv('WikiData.csv')
with open("wikidata.json", "w") as file:
    file.write(json.dumps(res, file, indent=4))
