from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
import sys
from pyspark import SparkConf, SparkContext
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from datetime import datetime


spark = SparkSession.builder.appName('ensemble_with_param_search').getOrCreate()
sc = spark.sparkContext

# Generates a hyperparameter RDD with the indicated parameters
def generateHyperParamsRDD() :
    # Use this for long tests
    # activationFuncs = ['logistic', 'tanh', 'relu']
    # learnRates = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001] # Learning Rates
    # maxIters = [50,100,200,500,1000,2000] # Max number of epochs
    # numHiddenL = [1,2,3] # Number of hidden layers
    # neuronsPerLayer = [1,2,5,10,20] # Number of neurons in each hidden layer
    # hiddenLayerNums = []

    # Use this for short tests
    activationFuncs = ['logistic']
    learnRates = [0.5,0.2,0.1,0.05,0.02] # Learning Rates
    maxIters = [500,1000,2000] # Max number of epochs
    numHiddenL = [1,2] # Number of hidden layers
    neuronsPerLayer = [1,2,5] # Number of neurons in each hidden layer
    hiddenLayerNums = []

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
 
schema = StructType([
    StructField('station', StringType(), False),
    StructField('dateofyear', IntegerType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
    StructField('label', IntegerType(), False),
])

input_file = sys.argv[1]

data = spark.read.csv(input_file,sep=' ',schema=schema)

data = data.na.drop()

assembler = VectorAssembler(inputCols=['dateofyear','latitude', 
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')

output = assembler.transform(data)

final_data = output.select('features','label')

train_data,test_data = final_data.randomSplit([0.7,0.3])

# FEATURE SELECTION -----------------------------------
# Generate the RDD of hyperparameters to test
paramsRdd = generateHyperParamsRDD()

# Get train and validation data
train,val = train_data.randomSplit([0.7,0.3])

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

# RDD with (model,accuracy)
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

# THIS IS WHERE HYPERPARAMETER SEARCH ENDS

trainer = MultilayerPerceptronClassifier(maxIter=iters, 
                                         layers=layers,
                                         stepSize=lr,
                                         blockSize=128, 
                                         seed=1234)

model = trainer.fit(train_data)

result = model.transform(test_data)

predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("Validation set accuracy = " + str(acc))
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
