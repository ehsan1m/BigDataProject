Our project has 5 important files. 

BoostingWithGridSearch.py
BoostingWithParameterSearch.py
ensembleWithParamGrid.py
ensembleWithParamSearch.py
scikitEnsemble.py

All of the files do the same thing with different methods . The ones with "Grid" in their names use the MLlib parameter grid search method to search for the best hyperparameter models, the ones with "ParameterSearch" are the ones that use the scikit-learn method to search for hyperparameters that we implemented. The "Boosting" ones use the boosting method to train and generate predictions, the "ensemble" ones use majority voting with 10 neural networks and "scikitEnsemble" does the same thing but with the scikit-learn method. The difference between the last 2 is that the first uses MLlib to create the neural networks and parallelizes each neural network separately and runs them sequentially, the second one parallelizes the whole ensemble, meaning that all 10 neural networks and being training or predicting at the same time rather than sequentially like in the MLlib version. 

The scikit-learn ensemble doesn't have a parameter grid version since that is unique to MLlib.

Keep in mind our project is divided in 2 parts, the parallelized hyperparameter search and the parallelized committee machine part. Each program shows results for both parts including runtime and model information for the best model obtained from the hyperparameter search and accuracy and confusion matrix for the final training/testing process for the comparison of committee machines. 

Each program is made to run with a small search space of hyperparameters to make them run faster. For larger test results please refer to the project report. 

As for the size of the datasets, we have a smaller one and a bigger one (weather2-tmax2 is small,weather3-tmax2 is bigger), but both are small enough to fit in memory, the reason for this is that we created these datasets by joining the weather and tmax datasets found in the cluster, but when trying to join the bigger versions, it would produce memory errors which caused us to be unable to generate bigger data sets. That said, in the report we explain why certain parts of the project should or should not be scalable and why.