import numpy
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.stats import mode

iris = load_iris()
X = iris.data
y = iris.target
#wczytaj dane

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
#dopasuj KMeans

labels_map = {}
for cluster in numpy.unique(y_kmeans):
    true_label = mode(y[y_kmeans == cluster], keepdims=False).mode
    labels_map[cluster] = true_label
    #najczesciej wystepująca rzeczywista etykieta w danym klastrze
#mapowanie klastrow na etykiety

y_kmeans_mapped = numpy.array([labels_map[cluster] for cluster in y_kmeans])
#zamiana etykiet klastrow na rzeczywiste etykiety

misclassified = numpy.sum(y_kmeans_mapped != y)
print(f"liczba punktow w niewlasciwym klastrze: {misclassified}")
print(f"procent punktow w niewlasciwym klastrze: {misclassified / len(y):.2%}")
#liczba blednie przypisanych punktow
