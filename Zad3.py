import matplotlib.pyplot as matplt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas
import itertools

iris = load_iris()
df = pandas.DataFrame(iris.data, columns=iris.feature_names)
features = iris.feature_names
pairs = list(itertools.combinations(features, 2))
#wczytanie danych

matplt.figure(figsize=(15, 10))
for i, (feat_x, feat_y) in enumerate(pairs):
    X_pair = df[[feat_x, feat_y]].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pair)
    #przygotowanie danych do klastrowania (tylko 2 wybrane cechy)
    matplt.subplot(2, 3, i + 1)
    matplt.scatter(
        X_pair[:, 0], X_pair[:, 1],
        c=clusters, cmap='viridis', alpha=0.7, edgecolor='k'
    )
    matplt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        c='red', s=100, marker='X', label='Centroidy'
    )
    matplt.xlabel(feat_x)
    matplt.ylabel(feat_y)
    matplt.title(f"KMeans: {feat_x} vs {feat_y}")
    if i == 0:
        matplt.legend()
matplt.tight_layout()
matplt.show()
#rysowanie wykresu
