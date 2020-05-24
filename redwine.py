import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
 
wine = pd.read_csv("winequality-red.csv", sep=";")
wine_except_quality = wine.drop("quality", axis=1)
X = wine_except_quality.values
y = wine['quality'].values

def LBFM(fset):
    A = np.ones((X.shape[0], 1))
    for f in fset:
        A = np.hstack((A, Sf(X)))
    X_train, X_test, y_train, y_test = train_test_split(A, y, random_state=0)
    w = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

    lst = []
    for x in X_test:
        lst.append(np.dot(x, w))
    y_pred = np.array(lst)

    return np.sqrt(mse(y_test, y_pred))


def poly(i):
    fset = []
    for j in range(1, i):
        fset.append(np.vectorize(lambda x: np.power(x, j)))
    return fset

def gauss(mu, sigma):
    return [np.vectorize(lambda x: np.exp(-(x - mu)**2 / (2*sigma**2)))]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LogisticSigmoid(mu, sigma):
    return [np.vectorize(lambda x: sigmoid((x - mu) / sigma))]


%evaluate with RSE
print("poly 1-dimentional", LBFM(poly(2)))
print("poly 2-dimentional", LBFM(poly(3)))
print("poly 3-dimentional", LBFM(poly(4)))
print("poly 4-dimentional", LBFM(poly(5)))
print("poly 5-dimentional", LBFM(poly(6)))
print("gaussian", LBFM(gauss(0,1))
print("logistic sigmoid", LBFM(LogisticSigmoid(0,1))
