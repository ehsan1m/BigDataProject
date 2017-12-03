import sys
import pandas as pd
import numpy as np
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

spark = SparkSession.builder.appName('ensemble_with_param_search').getOrCreate()
sc = spark.sparkContext

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+

# Generates a hyperparameter RDD with the indicated parameters
def generateHyperParamsRDD() :
    # Use this for long tests
    activationFuncs = ['logistic']
    learnRates = [0.2,0.1,0.05,0.02,0.01,0.005] # Learning Rates
    maxIters = [1000,1500] # Max number of epochs
    numHiddenL = [1,2,3] # Number of hidden layers
    neuronsPerLayer = [5,10,20] # Number of neurons in each hidden layer
    hiddenLayerNums = []
    
    # Use this for short tests
    # activationFuncs = ['logistic']
    # learnRates = [0.5,0.2,0.1,0.05,0.02] # Learning Rates
    # maxIters = [500,1000] # Max number of epochs
    # numHiddenL = [1] # Number of hidden layers
    # neuronsPerLayer = [1,2,5] # Number of neurons in each hidden layer
    # hiddenLayerNums = []

    # Fill in the different hidden layer neuron combinations
    for num in numHiddenL :
        for neu in neuronsPerLayer :
            neurons = [neu]
            if (num > 1) :
                for neu2 in neuronsPerLayer :
                    neurons += [neu2]
                    if (num > 2) :
                        for neu3 in neuronsPerLayer :
                            neurons += [neu3]
                            hiddenLayerNums.append(neurons)
                            neurons = neurons[:-1]
                    else :
                        hiddenLayerNums.append(neurons)
                        
                    neurons = neurons[:-1]
            else :
                hiddenLayerNums.append(neurons)

    # Fill in the RDD of hyperparameter combinations
    hyperParams = []
    for f in activationFuncs :
        for lr in learnRates :
            for iters in maxIters :
                for hl in hiddenLayerNums :
                    hyperParams.append([f,lr,iters,hl])


    # Transform the hyperparameter array into an RDD
    return sc.parallelize(hyperParams)

def listTransformTrain(row) :
    f = row.features
    return [f[0],f[1],f[2],f[3],f[4]]

def transformTest(row) :
    return row.label

def generateModels(params) :
    model =  MLPClassifier(solver='sgd', learning_rate='constant',
                    activation=params[0],
                    learning_rate_init=params[1],
                    max_iter=params[2],
                    hidden_layer_sizes=params[3])
    model.fit(rdd_train_X.value,rdd_train_y.value)
    preds = model.predict(rdd_val_X.value)
    return (model,accuracy_score(rdd_val_y.value,preds))

# 
def getBestModel(m1,m2) :
    if (m1[1] > m2[1]) :
        return m1
    else :
        return m2

# For a row with columns as predictions, choose the class based on the majority
def committee_voting(dataframe_row):
    total_values = dataframe_row.values.sum()
    if total_values >= (num_of_experts / 2):
        return 1
    else:
        return 0   

def genConfMatrix(realLabels,preds) :
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(realLabels)) :
        if (realLabels[i] == 1) and (preds[i] == 1) :
            TP += 1
        elif (realLabels[i] == 1) and (preds[i] == 0) :
            FN += 1
        elif (realLabels[i] == 0) and (preds[i] == 1) :
            FP += 1
        elif (realLabels[i] == 0) and (preds[i] == 0) :
            TN += 1

    return TP,TN,FP,FN                
 
schema = StructType([
    StructField('station', StringType(), False),
    StructField('dateofyear', FloatType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
    StructField('label', IntegerType(), False),
])

input_file = sys.argv[1]

print("Reading input data...")
data = spark.read.csv(input_file,sep=' ',schema=schema)

data = data.na.drop()

assembler = VectorAssembler(inputCols=['dateofyear','latitude', 
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')

output = assembler.transform(data)

processed_data = output.select('features','label')

train_data,test_data = processed_data.randomSplit([0.8,0.2])

# FEATURE SELECTION -----------------------------------
# Generate the RDD of hyperparameters to test
print("Filling Hyperparameter RDD...")
paramsRdd = generateHyperParamsRDD()

# Get train and validation data
train,val = train_data.randomSplit([0.8,0.2])

# Generate dataframes for classlabels and drop class rows
trainY = train.select(train.label)
valY = val.select(val.label)
train = train.drop(train.label)
val = val.drop(val.label)

# Transform dataframes into rdds
trainRdd = train.rdd 
valRdd = val.rdd 
trainYRdd = trainY.rdd 
valYRdd = valY.rdd

# Generate train and test Rdds with rows as lists
trainRdd = trainRdd.map(listTransformTrain)
valRdd = valRdd.map(listTransformTrain)

# Generate train and val class labels as singletons
trainYRdd = trainYRdd.map(transformTest)
valYRdd = valYRdd.map(transformTest)

# Broadcast train data and train labels
rdd_train_X = sc.broadcast(trainRdd.collect())
rdd_train_y = sc.broadcast(trainYRdd.collect())

# # Broadcast test data
rdd_val_X = sc.broadcast(valRdd.collect())
rdd_val_y = sc.broadcast(valYRdd.collect())

# Measure time
start = time.time()

# RDD with (model,accuracy)
print("Calculating best model...")
modelsRdd = paramsRdd.map(generateModels)

# Get the model with best accuracy :
bestModel = modelsRdd.reduce(getBestModel)

# Put hyperparams into variables
act = bestModel[0].activation
iters = bestModel[0].max_iter
lr = bestModel[0].learning_rate_init
hlayers = bestModel[0].hidden_layer_sizes
acc = bestModel[1]

layers = [5] + hlayers + [2]

end = time.time()

print("---------------------- Best model info ----------------------")
print("Activation func : "+act)
print("Max epochs : "+str(iters))
print("Learning rate : "+str(lr))
print("Hidden layers : " + str(hlayers))
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------")
# THIS IS WHERE HYPERPARAMETER SEARCH ENDS

# Define number of experts (neural nets) to be trained
num_of_experts = 10

# Dictionary to store the models trained for each expert
dict_of_models = dict()

# List of dataframes for each of the experts
dataframes = train_data.randomSplit([1.0 for x in range(num_of_experts)],seed=1234)

# Get the models for each expert using the parameters of the best model defined above
print("Generating and training experts...")
start = time.time()
for expert in range(num_of_experts):

    train_data_experts,test_data_experts = dataframes[expert].randomSplit([0.8,0.2])
    
    trainer = MultilayerPerceptronClassifier(maxIter=iters, 
                                             layers=layers,
                                             stepSize=lr,
                                             blockSize=128, 
                                             seed=1234)
    model = trainer.fit(train_data_experts)
    dict_of_models[expert] = model

# Dictionary to store the predictions of the full dataset for each trained expert
dict_of_predictions = dict()

# Iterate through the expert and predict the values of each dataset
print("Generating predictions...")
for expert in range(num_of_experts):
    dict_of_predictions[expert] = dict_of_models[expert].transform(test_data)

# Create a pandas dataframe whose columns are each predictions of each expert
evaluations = pd.concat([dict_of_predictions[x].toPandas().prediction for x in range(num_of_experts)],axis=1)

# Rename the prediction columns to reference the number of the expert (sequential)
evaluations.columns = ['prediction'+str(x) for x in range(num_of_experts)]

# Create a list with the result of the committee votation
evaluations_vote = []
for index, row in evaluations.iterrows():
    evaluations_vote.append(committee_voting(row))

end = time.time()

# Create a dataframe with the label and the prediction
predictionAndLabels = pd.concat([test_data.toPandas().label, pd.DataFrame(evaluations_vote)],axis=1)
predictionAndLabels.columns = ['label','predictions']

# Create tha colum 'evaluation' to check if the prediction is correct
predictionAndLabels['evaluation'] = np.where(predictionAndLabels.label == predictionAndLabels.predictions, 1, 0)

accuracy = predictionAndLabels.evaluation.sum() / predictionAndLabels.shape[0]

TP,TN,FP,FN= genConfMatrix(predictionAndLabels.label, predictionAndLabels.predictions)

print("---------------------- Final train/test info ----------------------")
print("Final accuracy : "+str(accuracy))
print("Real 1s classified as 1s : "+str(TP)) 
print("Real 0s classified as 0s : "+str(TN)) 
print("Real 1s classified as 0s : "+str(FN))
print("Real 0s classified as 1s : "+str(FP))
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------------")

