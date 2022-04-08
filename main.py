import pandas as pd


def drop_nan(data):
    print(data)
    print(data.isna().sum())
    # count_nan = data['sepal_length'].isna().sum()
    # print('Count of NaN: ' + str(count_nan))
    # print(data.isnull().groupby(data['sepal_length', 'sepal_width', 'petal_length', 'petal_width']).sum().sum(axis=1))

    return data.dropna()


def run_program(name):
    data = load_data(name)
    data = drop_nan(data)


def load_data(name):
    return pd.read_csv(name,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


if __name__ == '__main__':
    run_program('iris.data')
