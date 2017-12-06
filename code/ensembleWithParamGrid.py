import pandas as pd
import numpy as np
import sys
import time
import itertools

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

spark = SparkSession.builder.appName('ensemble_with_param_grid').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+

# For a row with columns as predictions, choose the class based on the majority
def committee_voting(dataframe_row):
    total_values = dataframe_row.values.sum()
    if total_values >= (num_of_experts / 2):
        return 1
    else:
        return 0 

# Generate a permutation of all elements of the list input_layers and concatenate
# to it's beginning and end the number of neurons in the input and output layers
def generateLayersCombination(hidden_layers, input_layer, output_layer):
    layers_combination = []
    for i in range(len(hidden_layers)+1):
        for j in list(list(tup) for tup in itertools.permutations(hidden_layers, i)):
            layers_combination.append(j)

    for i in range(len(layers_combination)):
        layers_combination[i] = input_layer + layers_combination[i] + output_layer

    return layers_combination

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

#drop the rows with 'nan' values (if exist)
data = data.na.drop()

# create a column with the composition of all the features
assembler = VectorAssembler(inputCols=['dateofyear','latitude', 
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')
output = assembler.transform(data)

# create spark df with the columns features and label
processed_data = output.select('features','label')

train_data,test_data = processed_data.randomSplit([0.8,0.2])

##### GRID PARAMETER BUILDER

mlpc = MultilayerPerceptronClassifier(blockSize=128, seed=1234)


    # Use this for long tests
    # activationFuncs = ['logistic']
    # learnRates = [0.2,0.1,0.05,0.02,0.01,0.005] # Learning Rates
    # maxIters = [1000,1500] # Max number of epochs
    # numHiddenL = [1,2,3] # Number of hidden layers
    # neuronsPerLayer = [5,10,20] # Number of neurons in each hidden layer
    # hiddenLayerNums = []

print("Creating parameter grid builder...")
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
# paramGrid = ParamGridBuilder() \
#     .addGrid(mlpc.maxIter, [1000,1500]) \
#     .addGrid(mlpc.layers, generateLayersCombination(hidden_layers = [5,10,20], input_layer = [5], output_layer = [2])) \
#     .addGrid(mlpc.stepSize, [0.2,0.1,0.05,0.02,0.01,0.005])\
#     .build()

# SIMPLER COMBINATION FOR TEST
paramGrid = ParamGridBuilder() \
    .addGrid(mlpc.maxIter, [500,1000]) \
    .addGrid(mlpc.layers, generateLayersCombination(hidden_layers = [1,2,5], input_layer = [5], output_layer = [2])) \
    .addGrid(mlpc.stepSize, [0.5,0.2,0.1,0.05,0.02]) \
    .build()


print("Calculating best model...")
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
start = time.time()
tvs = TrainValidationSplit(estimator=mlpc,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train_data)

# Save the parameters of the best model into variables
bestmodel = model.bestModel
layers = list(bestmodel._java_obj.parent().getLayers())
iters = bestmodel._java_obj.parent().getMaxIter()
# solver = bestmodel._java_obj.parent().getSolver()
# tol = bestmodel._java_obj.parent().getTol()
lr = bestmodel._java_obj.parent().getStepSize()

end = time.time()

print("---------------------- Best model info ----------------------")
print("Max epochs : "+str(iters))
print("Learning rate : "+str(lr))
print("Layers : " + str(layers))
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------")

best_model_time = end - start

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
                                             # solver=solver, ##### include in the other version
                                             # tol=tol, #### include in the other version
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

TP,TN,FP,FN = genConfMatrix(predictionAndLabels.label, predictionAndLabels.predictions)

print("---------------------- Final train/test info ----------------------")
print("Final accuracy : "+str(accuracy))
print("Real 1s classified as 1s : "+str(TP)) 
print("Real 0s classified as 0s : "+str(TN)) 
print("Real 1s classified as 0s : "+str(FN))
print("Real 0s classified as 1s : "+str(FP))
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------------")

result = {"accuracy": [accuracy], "best_model_time": [best_model_time], "prediction_time": [(end - start)] , "TP": [TP], "TN": [TN], "FP": [FP], "FN": [FN]}
output = spark.createDataFrame(pd.DataFrame.from_dict(result))
output.coalesce(1).write.csv('output-bigdata-project/',mode='overwrite',header=True)
