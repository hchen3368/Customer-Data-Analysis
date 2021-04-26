# import necessary modules
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def Perform_PCA(df, n_components = 1):
    """
    INPUT:

    df : a pd.dataframe, data for PCA
    n_component: an integer, defines the number of component in PCA

    OUTPUT:

    Return a tuple (df_components, pca), where
    df_components: a dataframe saving the components of the PCA
    pca: a sklearn PCA instance

    Also plot the explained variance and print the total explained variance.
    """



    # Standardize the data
    df_standard = StandardScaler().fit_transform(df)

    # instantiate PCA and fit_transform
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_standard)

    # Plot the explained variance of the components
    plt.bar(range(n_components), pca.explained_variance_ratio_)
    plt.xlabel('Component')
    plt.xticks(range(n_components))
    plt.ylabel('Explained Variance')
    plt.title(f'PCA Explained Variance, n_components={n_components}')
    plt.show()

    print(f'The total explained variance by the {n_components} components is {sum(pca.explained_variance_ratio_)}')

    # Save components to a DataFrame
    df_components = pd.DataFrame(components)

    return df_components, pca


# a function to perform grid search to find best hyperparameter(s)
def grid_search_param(model, parameters, X, y, n_jobs=1, n_folds=5, scoring=None):

    """
    This function perform gird search to find the best hyperparameter(s) for a given model,
    using n-fold cross validation.


    INPUT:
    model: a sklearn classifer
    parameters: np.array like, the set of parameters to try
    X: np.array, variable data for the model
    y: np.array, targets for the model
    n_jobs: number of job to be parallelized, default 1.
    n_folds: number of folds for cross-validation splits, default 5.
    scoring: scoring function for the validation process, default None. If None, use model default function.


    OUTPUT:
    Return the best estimator.
    Print out the best parameters, best score, and grid scores.

    """

    if scoring:
        searcher = GridSearchCV(model, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, scoring=scoring)
    else:
        searcher = GridSearchCV(model, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)

    searcher.fit(X, y)
    print("The (best_params, best_score) in GridSearchCV are:", searcher.best_params_, searcher.best_score_)


    return searcher.best_estimator_


def train_tune_score(model, parameters, data, target, upsample=False, scoring=None, n_folds=5, n_jobs=1):
    """
    This function does:

    1. use n-fold cross validation on training set to determine the best hyperparameter(s),
    2. fit model with the training set,
    3. then reports model scores, f1, recall, and confusion matrix, return trained model.

    INPUT:
    model: a sklearn classifer.
    parameters: np.array like, the set of parameters to try.
    data: a 4-tuple of np.arrays (X_train, X_test, y_train, y_test), consists of the train and test data.
    target: the column name of the target column.
    upsample: boolean, if True, will perform upsampling of the training data.
    n_jobs: number of job to be parallelized, default 1.
    n_folds: number of folds for cross-validation splits, default 5.
    scoring: scoring function for the validation process, default None. If None, use model default function.

    OUTPUT:
    return the trained best model, and train set, test set.
    print out scores on train and test sets, print out confusion_matrix.

    """

    # train_test split
    X_train, X_test, y_train, y_test = data

    if parameters:
        model = grid_search_param(model, parameters, X_train, y_train, n_jobs=n_jobs, n_folds=n_folds, scoring=scoring)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f'(train_accuracy, test_accuracy) are ({train_accuracy},{test_accuracy}).')
    print(f"F1_score on test set is: {f1_score(y_test, y_pred)}.")
    print(f"Recall on test data:{recall_score(y_test, y_pred)}.")
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model
