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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f}'.format(rmse))
    print('Mean Absolute Error: {:0.4f}'.format(mae))
    print('R2 Score: {:0.4f}'.format(r2))
    metrics = np.array([rmse, mae, r2])
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

    LASSO = Lasso(alpha = 0.075)
    LASSO.fit(xTrain, yTrain)
    yPred = LASSO.predict(xTest)
    return yPred


def RidgePred(xTrain, yTrain, xTest, idx):
    '''
    Makes a prediction after fitting the model to the training data
    Inputs: 2D Numpy Array, 1D Numpy Array, 2D Numpy Array, 1D Numpy Array
    Outputs: 1D Numpy Array, 1D Numpy Array
    '''
    alphas = np.array([192.7, 548.3, 122.3, 61.1, 128.3, 128.4, 306.5, 2000.0, 177.6, 381.6, 404.3, 57.7, 259.9, 205.9, 560.7, 196.8, 79.2, 171.2, 315.8, 451.8, 490.6, 213.3, 156.7, 2000.0])
    ridge = Ridge(alpha=alphas[idx], random_state=42)
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


def KFoldCV(X, y, reg, n_splits=5, idx=0):
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
            y_pred = RidgePred(X_train, y_train, X_test, idx)
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


def predVsActual(predicted, actual, reg, drug, save=False):
    sns.scatterplot(actual, predicted, color = 'darkmagenta')
    sns.despine()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual ' + reg + ' ' + drug, size=15)
    plt.xlim((0,6))
    plt.ylim((0,6))
    if save:
        plt.savefig(reg + '.png', dpi = 1000, bbox_inches = "tight")
    plt.show()
