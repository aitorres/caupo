import time

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from caupo.preprocessing import preprocess_v1
from caupo.utils import get_text_from_all_tweets, plot_clusters

all_tweets = get_text_from_all_tweets()
tweets_subset = get_text_from_all_tweets(
    city="Caracas",
    dates=[
        "2021-02-15",
        "2021-02-16",
        "2021-02-17",
        "2021-02-18",
        "2021-02-19",
        "2021-02-20",
        "2021-02-21",
    ])

print("All tweets: ", len(all_tweets))
print("Subset: ", len(tweets_subset))

all_preprocessed = list(set(preprocess_v1(all_tweets)))
subset_preprocessed = list(set(preprocess_v1(tweets_subset)))

print("All tweets prepro: ", len(all_preprocessed))
print("Subset prepro: ", len(subset_preprocessed))

tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(all_preprocessed)]
doc2vec_model = Doc2Vec(tagged_documents, vector_size=150, window=5, min_count=3, workers=8)

vectors = [doc2vec_model.infer_vector(doc.split()) for doc in subset_preprocessed]

scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(vectors)

pca_fit = PCA(n_components=2)
scatterplot_vectors = pca_fit.fit_transform(scaled_vectors)

K_VALUES = (2, 3, 4, 5, 6)
for k in K_VALUES:
    print(f"k={k}")
    km = KMeans(n_clusters=k)
    t0 = time.time()
    km_result = km.fit(scaled_vectors)
    t1 = time.time()
    labels = km_result.labels_
    inertia = km.inertia_
    print("Calculating metrics")
    sil_score = silhouette_score(scaled_vectors, labels, metric='euclidean')
    print("Silhouette: ", sil_score)
    cal_har_score = calinski_harabasz_score(scaled_vectors, labels)
    print("CH: ", cal_har_score)
    dav_boul_score = davies_bouldin_score(scaled_vectors, labels)
    print("DB: ", dav_boul_score)

    plot_clusters(scatterplot_vectors,
                  filename=f"quick_clusters_{k}.png",
                  title=f'Clusters Representation (k={k}) for `Doc2Vec`',
                  labels=labels)

