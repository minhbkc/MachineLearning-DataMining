from random import seed
import numpy as np
from dataloaders import fish
from models import knn
from sklearn.neighbors import KNeighborsClassifier


seed(1)
filename = '/home/genkibaskervillge/Documents/Hust/MachineLearning_DataMining/fistProject/dat/Fish.csv'
X_train, X_test, y_train, y_test = fish.get(ratio=0.3, file_path=filename)
print('Training size:{}'.format(X_train.shape))
# print(y_train)
print('Test size:{}'.format(X_test.shape))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# evaluate algorithm
# Model training
bestK = 0
bestAcc = 0
for ratio in range(1, 20):
    print("Number of neighbours:{}".format(ratio))
    model = knn.K_Nearest_Neighbors_Classifier(K=ratio)
    model.fit(X_train, y_train)
    model1 = KNeighborsClassifier(n_neighbors=ratio)
    model1.fit(X_train, y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)
    Y_pred1 = model1.predict(X_test)
    # print('Y_pred1:{}'.format(type(Y_pred1)))
    # print('y_test:{}'.format(type(y_test)))

    # measure performance
    correctly_classified = 0
    correctly_classified1 = 0

    # counter
    count = 0
    for count in range(np.size(Y_pred1)):
        if y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1
        if y_test[count] == Y_pred1[count]:
            correctly_classified1 = correctly_classified1 + 1
        count = count + 1

    print("Accuracy on test set by our model:", (correctly_classified / count) * 100)
    print("Accuracy on test set by sklearn model:", (correctly_classified1 / count) * 100)
    print("\n")

    if bestAcc < ((correctly_classified / count) * 100):
        bestAcc = ((correctly_classified / count) * 100)
        bestK = ratio

print("Best neighbors:{}, with best accuracy is:{}".format(bestK, bestAcc))
