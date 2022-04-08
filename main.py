import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(name):
    return pd.read_csv(name,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


def drop_nan(data):
    print("\n", data.isna().sum())
    data.dropna(inplace=True)
    # This line screwed me for hours. When dropping, it doesn't recalculate index
    # drop = True means "don't keep old index" (which, for some reason it does after "RECALCULATING" index)
    data.reset_index(drop=True, inplace=True)
    return data


def encode_data(data):
    data['target'] = preprocessing.LabelEncoder().fit_transform(data['target'])
    return data


def normalize_data(data):
    mean = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variance = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()

    target = data.pop('target')
    std_data = preprocessing.StandardScaler().fit_transform(data)
    std_data = pd.DataFrame(std_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    mean2 = std_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variance2 = std_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()

    print("\n\nVariances:\n{}".format(pd.concat({'before': variance, 'after': variance2}, axis=1)))
    print("\n\nMeans:\n{}".format(pd.concat({'before': mean, 'after': mean2}, axis=1)))

    return std_data, target


def pca_data(data):
    principal = PCA(n_components=2)
    principal.fit(data)
    return pd.DataFrame(principal.transform(data), columns=['D1', 'D2'])


def plot_data(data, data_2d):

    Iris_setosa = []
    Iris_versicolor = []
    Iris_virginica = []
    for i in range(len(data_2d)):
        if data.iloc[i]['target'] == 0:
            Iris_setosa.append(data_2d[i])
        elif data.iloc[i]['target'] == 1:
            Iris_versicolor.append(data_2d[i])
        else:
            Iris_virginica.append(data_2d[i])

    plt.scatter([i[0] for i in Iris_setosa], [i[1] for i in Iris_setosa])
    plt.scatter([i[0] for i in Iris_versicolor], [i[1] for i in Iris_versicolor])
    plt.scatter([i[0] for i in Iris_virginica], [i[1] for i in Iris_virginica])
    plt.legend(["setosa", "versicolor", "virginica"])
    plt.show()

    data.drop(['target'])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)
    plt.show()


def run_program(name):
    data = load_data(name)
    data = drop_nan(data)
    data = encode_data(data)
    normalized_data, target = normalize_data(data)
    data_2d = pca_data(normalized_data)

    data_2d['color'] = target.map({0: 'Red', 1: 'Blue', 2: 'Black'})
    ax = data_2d.plot.scatter(x="D1", y="D2", c="color")
    plt.show()


if __name__ == '__main__':
    run_program('iris.data')
