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

N = X_train[0] # number of training sample 
d = X_train[1] # data dimension 
C = 7 # number of classes 


def reshape_data(data):
    re = np.zeros((data.shape[1], data.shape[0]))
   
    for idx, data in enumerate(data):
        re[:, idx] = data
   
    return re.astype(np.float32)

X_train = reshape_data(X_train)
X_test = reshape_data(X_test)
# y_train = y_train.reshape(y_train.shape[1], y_train.shape[0])
# y_test = y_test.reshape(y_test.shape[1], y_test.shape[0])

from scipy import sparse 
def convert_labels(y, C = C):
    """
    convert 1d label to a matrix label: each column of this 
    matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, 
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

Y_train = convert_labels(y_train, C)
Y_test = convert_labels(y_test, C)

def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A

# cost or loss function  
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

W_init = np.random.randn(d, C)

def grad(X, Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)
    
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)
    return g 

g1 = grad(X_train, Y_train, W_init)
g2 = numerical_grad(X_train, Y_train, W_init, cost)

print(np.linalg.norm(g1 - g2))

def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]    
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W
eta = .05 
d = X_train.shape[0]
W_init = np.random.randn(d, C)

W = softmax_regression(X_train, Y_train, W_init, eta)
print(W[-1])

def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)

y_pred = pred(W_init, X_test)
