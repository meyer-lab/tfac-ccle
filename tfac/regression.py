"""Performs regression on the drug data and cell line factors"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


path = os.path.dirname(os.path.abspath(__file__))


def errorMetrics(y_test, y_pred):
    '''
    Determines error values based off of predicted and actual values
    Inputs: 1D Numpy Arrays
    Outputs: Prints rmse and r2  and returns them as Float64
    '''
    weightedRes = (y_pred - y_test) / y_test
    absError = abs(weightedRes) * 100
    sqError = (weightedRes**2) * 100

    mape = np.round(np.mean(absError), 2)
    mspe = np.round(np.mean(sqError), 2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f}'.format(rmse))
    print('Mean Absolute Error: {:0.4f}'.format(mae))
    print('Accuracy (MSPE): {:0.2f}'.format(100 - mspe))
    print('Accuracy (MAPE): {:0.2f}'.format(100 - mape))
    print('R2 Score: {:0.4f}'.format(r2))
    metrics = np.array([rmse, mae, mspe, mape, r2])
    return metrics


def OLSPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    OLS = LinearRegression()
    OLS.fit(xTrain, yTrain)
    yPred = OLS.predict(xTest)
    return yPred


def LASSOPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data

    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array

    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    LASSO = Lasso(alpha=0.075, random_state=42)
    LASSO.fit(xTrain, yTrain)
    yPred = LASSO.predict(xTest)
    return yPred


def RidgePred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    ridge = Ridge(alpha=122.358, random_state=42)
    ridge.fit(xTrain, yTrain)
    yPred = ridge.predict(xTest)
    return yPred


def ElasticNetPred(xTrain, yTrain, xTest):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''

    elasticNet = ElasticNet(alpha=0.59, l1_ratio=0.031)
    elasticNet.fit(xTrain, yTrain)
    yPred = elasticNet.predict(xTest)
    return yPred


def KFoldCV(X, y, reg, n_splits=5):
    '''Performs KFold Cross Validation on data'''
    kfold = KFold(n_splits, True, 19)
    y_pred = 0
    yPredicted = 0
    yActual = 0
    for rep, indices in enumerate(kfold.split(X)):
        X_train, X_test = X[indices[0]], X[indices[1]]
        y_train, y_test = y[indices[0]], y[indices[1]]
        if reg == 'OLS':
            y_pred = OLSPred(X_train, y_train, X_test)
        elif reg == 'LASSO':
            y_pred = LASSOPred(X_train, y_train, X_test)
        elif reg == 'Ridge':
            y_pred = RidgePred(X_train, y_train, X_test)
        elif reg == 'ENet':
            y_pred = ElasticNetPred(X_train, y_train, X_test)

        if rep == 0:
            yPredicted = y_pred
            yActual = y_test
        else:
            yPredicted = np.concatenate((yPredicted, y_pred))
            yActual = np.concatenate((yActual, y_test))
        r2 = r2_score(yActual, yPredicted)
    return r2, yPredicted, yActual
