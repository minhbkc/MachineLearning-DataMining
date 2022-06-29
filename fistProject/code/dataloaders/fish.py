import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def get(ratio=0.3, file_path=''):
    # file_path = '/home/genkibaskervillge/Documents/Hust/MachineLearning_DataMining/fistProject/dat/Fish.csv'
    data = pd.read_csv(file_path)
    print('Data:', data.head(5))

    # dealing with duplicate data
    data.drop_duplicates(keep=False, inplace=True)

    # convert categorical data numerical data
    le = LabelEncoder()
    label = le.fit_transform(data['Species'])
    print('Label:', label)
    data.drop("Species", axis=1, inplace=True)
    data["Species"] = label

    # separate label and data feature
    y = data['Species']
    X = data.drop('Species', axis=1)

    # train, test split
    print('Train test split...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)

    # normalize
    print('Normalize...')
    norm = StandardScaler()
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.fit_transform(X_test)

    return X_train, X_test, y_train, y_test
