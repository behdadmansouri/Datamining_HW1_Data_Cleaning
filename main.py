import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_data(file_location):
    """
    reads the data from our .data file (which is basically .csv)
    :param file_location: the location of the file
    :return: the contents of the file as a pandas dataframe
    """
    return pd.read_csv(file_location,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


def drop_nan(data):
    """
    prints how many NaN we have per column
    drops the NaN rows from our dataframe
    :param data: the data
    :return: the data, cleaned from NaN
    """
    print("\n", data.isna().sum())
    data.dropna(inplace=True)
    # This line screwed me for hours. When dropping, it doesn't recalculate index
    # drop = True means "don't keep old index" (which, for some reason it does after "RECALCULATING" index)
    data.reset_index(drop=True, inplace=True)
    return data


def encode_data(data):
    """
    encodes the categorical data into int
    :param data: the data
    :return: the data, where the 'target' column strings are replaced with numbers
    """
    data['target'] = preprocessing.LabelEncoder().fit_transform(data['target'])
    return data


def normalize_data(data):
    """
    pops the 'target' column from our data
    normalizes the data
    and prints how the mean and the variance changed for the data
    :param data: the data
    :return: data, without 'target' but normalized, along with target (now a pandas dataframe of its own)
    """
    target = data.pop('target')

    mean = data.mean()
    variance = data.var()

    std_data = preprocessing.StandardScaler().fit_transform(data)
    std_data = pd.DataFrame(std_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    mean2 = std_data.mean()
    variance2 = std_data.var()

    print("\n\nVariances:\n{}".format(pd.concat({'before': variance, 'after': variance2}, axis=1)))
    print("\n\nMeans:\n{}".format(pd.concat({'before': mean, 'after': mean2}, axis=1)))

    return std_data, target


def pca_data(data):
    """
    reduces the features of our data from 4 to 2 dimensions
    :param data: the data
    :return: the data, now in two dimensions 'D1' and 'D2'
    """
    principal = PCA(n_components=2)
    principal.fit(data)
    return pd.DataFrame(principal.transform(data), columns=['D1', 'D2'])


def plot_data(data, data_2d, target):
    """
    shows a scatter plot of our data points
    X and Y axis being the extracted features and the dots are colored by the corresponding 'target'
    then, shows a box plot of the data columns - the initial features
    :param data: the data
    :param data_2d: the data, normalized and 2D with new extracted features
    :param target: the 'target' column we had initially, now about to merge with "data_2d" so we know the rows' origins
    """
    data_2d['color'] = target.map({0: 'Red', 1: 'Blue', 2: 'Black'})
    data_2d.plot.scatter(x="D1", y="D2", c="color")
    plt.show()
    data.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    plt.show()


def run_program(file_location):
    """
    runs the program
    by first loading the data
    then dropping any row with a NaN entry
    next up, it encodes the categorical data
    and then normalizes the range of our data
    after that, two new features will be extracted from the 4 we initially had
    and finally we plot some charts
    :param file_location: the location of our data file
    """
    data = load_data(file_location)
    data = drop_nan(data)
    data = encode_data(data)
    normalized_data, target = normalize_data(data)
    data_2d = pca_data(normalized_data)
    plot_data(data, data_2d, target)


if __name__ == '__main__':
    run_program('iris.data')
