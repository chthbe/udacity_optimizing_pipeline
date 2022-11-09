# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This problem is started with a banking marketing dataset and want to use classification algorithms to predict a success indicator based on customer data. Utilising the AzureML python SDK the data set is generated and then consumed in 2 different ways: a more manual logistic regression with hyperdrive optimized parameters and a fullon automl based approach.

While the 2 model outcomes seem close in accuracy with 91.6% (logistic regression) and 91.8% (AutoML), they are not comparable as the logistic regression uses a test set and the AutoML uses cross validation. The cross validation accuracy is more reliable as it can happen that by selecting the best test-set accuracy, the model overfits the test set.
With the accuracy being better and more reliable, the AutoML outperforms the manual approach.

## Scikit-learn Pipeline
The scikit learn pipeline employs a simple training script gathering and cleaning the data, transforming it to pandas and then running a default logistic regression from the scikit learn package utilising parsed arguments for the regularisation parameter C and the maximum iterations parameter. This is then embedded in a scriptrunconfig allowing to utlise Hyperdrive to optimise the primary metric, which in this case was chosen to be accuracy.
The parameter of maximum iterations (max_iter) is an integer value mostly used to avoid running slow convergence iterations a long time. As such here a choice slelction was used chosing between 50, 100, 150 and 200 iterations. These values were guesstimates.
The parameter C is a regularisation parameter which regularises stronger the smaller it is and is a floating point variable. The values here were chosen at random using a uniform distribution between 0.5 and 2.0 in order not to sway too far from the default parameter of 1.0.
To select the pairs of parameters a random parameter sampling approach was used to reduce overall runtime compared to a more extensive search.
A simple bandit policy was chosen to use for early stopping as it supports early stopping when no further improvement is detected while still being stable against negative outliers.
The returned optimized model had an accuracy of 91.6% on the training set which was drawn at random and made up 25% of the entire set. The optimal parameters were 1.94 for C and 150 for max iterations.

## AutoML
The autoML pipeline starts with cleaning the data the same way as the manual train script did and registering it as a dataset. Afterwards, a configuration for the autoML is prepared utilising a 3 fold cross validation, giving the full training data and specify the classification task and variable. In order to keep runtime in check a model timeout is defined at 30 minutes. The model is registered as a new experiment and subsequently run.
The optimal output from the automl is a voting ensemble based on the best previously created models and reaches an cross-validation accuracy of 91.8%. 

## Pipeline comparison
The manual model is more complex in setup and less optimised. The data is transformed and cleaned in every run through hyperdrive. The train-test-set setup reduces runtime but makes the result more biased and thus leading to potentially worse generelisation. This run also focuses solely on getting the best out of one specific model.
The automl run is easily setup and started and relies on only one cleaning and registration step of the data. Additionally to testing several models, it also tests different normalisation scalers. Thanks to being cross-validation based, it gives a good read on which models work well for the given data.
If one model works clearly superior to others in the autoML run it can be beneficial to run a dedicated parameter tuning run like the manual run to optimise the chosen model further.

## Future work
Some improvements for future models include:
- Implementing cross-validation for the manual run
- Utilising sampling or weighting techniques to balance the classes as only around 11% of the data represent the positive class
- A different metric from accuracy should be considered due to the unbalanced dataset, but this is depending on the model purpose and business context
