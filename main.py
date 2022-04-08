import pandas as pd
from sklearn import preprocessing

def drop_nan(data):
    print(data.isna().sum())
    return data.dropna()


def encode_data(data):
    data['target'] = preprocessing.LabelEncoder().fit_transform(data['target'])
    return data


def normalize_data(data):
    averages = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variances = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()

    target_temp = data.drop(['target'], axis=1, inplace=True)
    standardized_data = preprocessing.StandardScaler().fit_transform(data)
    standardized_data = pd.DataFrame(standardized_data, columns=[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
    standardized_data['target'] = target_temp

    averages2 = standardized_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].mean()
    variances2 = standardized_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].var()
    for i in range(4):
        print("column %2d" % (i + 1))
        print("variance : %5.2f --> %5.2f" % (averages[i], averages2[i]))
        print("mean     : %5.2f --> %5.2f" % (variances[i], variances2[i]))
    return standardized_data

def run_program(name):
    data = load_data(name)
    data = drop_nan(data)
    data = encode_data(data)
    print(data)
    data = normalize_data(data)
    print(data)

def load_data(name):
    return pd.read_csv(name,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


if __name__ == '__main__':
    run_program('iris.data')
