import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(name):
    return pd.read_csv(name,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


def drop_nan(data):
    print(data.isna().sum())
    return data.dropna()


def encode_data(data):
    data['target'] = preprocessing.LabelEncoder().fit_transform(data['target'])
    return data


def normalize_data(data):
    mean = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variance = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()

    data.pop('target')
    std_data = preprocessing.StandardScaler().fit_transform(data)
    std_data = pd.DataFrame(std_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    mean2 = std_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variance2 = std_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()

    print("\n\nVariances:{}".format(pd.concat({'before': variance, 'after': variance2}, axis=1)))
    print("\n\nmeans:{}".format(pd.concat({'before': mean, 'after': mean2}, axis=1)))

    return std_data


def pca_data(data):
    principal = PCA(n_components=2)
    principal.fit(data)
    return principal.transform(data)


def plot_data(data, data_2d):
    Iris_setosa = []
    Iris_versicolor = []
    Iris_virginica = []
    for i in range(len(data_2d)):
        if data_2d.iloc[i]['target'] == 0:
            Iris_setosa.append(data_2d[i])
        elif data_2d.iloc[i]['target'] == 1:
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
    data = normalize_data(data)
    data_2d = pca_data(data)
    plot_data(data, data_2d)


if __name__ == '__main__':
    run_program('iris.data')
