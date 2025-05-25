import matplotlib.pyplot as plt
import seaborn
from sklearn.datasets import load_iris
import pandas
import itertools

iris = load_iris()
df = pandas.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pandas.Categorical.from_codes(iris.target, iris.target_names)
#wczytanie danych

features = iris.feature_names
pairs = list(itertools.combinations(features, 2))
#mozliwe pary cech

palette = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
markers = {'setosa': 'o', 'versicolor': 's', 'virginica': 'D'}
#mapowanie gatunkow na kolory i symbole

plt.figure(figsize=(15, 10))
for i, (feat_x, feat_y) in enumerate(pairs):
    plt.subplot(2, 3, i+1)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.scatter(
            subset[feat_x],
            subset[feat_y],
            label=species if i == 0 else "",
            c=palette[species],
            marker=markers[species],
            alpha=0.7,
            edgecolor='k'
        )
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.title(f"{feat_x} vs {feat_y}")
plt.legend()
plt.tight_layout()
plt.show()

##Zad2. Najlatwiej rozdzielic gatunki na wykresach z cechami platka (petal), a najtrudniej — z cechami dzialki kielicha (sepal),
#zwlaszcza sepal width. Potwierdzaja to wykresy ze zbioru Iris.
