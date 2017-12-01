from pyspark import SparkConf, SparkContext
import sys
import operator
import math
import re, string
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from datetime import datetime

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

 
input_file = sys.argv[1]

spark = SparkSession.builder.appName('Boosting with parameter search').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+
 
# Generates a hyperparameter RDD with the indicated parameters
def generateHyperParamsRDD() :
	# Use this for long tests
	# activationFuncs = ['logistic']
	# learnRates = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001] # Learning Rates
	# maxIters = [500,1000,2000] # Max number of epochs
	# numHiddenL = [1,2,3] # Number of hidden layers
	# neuronsPerLayer = [2,5,10,20] # Number of neurons in each hidden layer
	# hiddenLayerNums = []

	# Use this for short tests
	activationFuncs = ['logistic']
	learnRates = [0.5,0.2,0.1,0.05,0.02] # Learning Rates
	maxIters = [500,1000] # Max number of epochs
	numHiddenL = [1] # Number of hidden layers
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
	return [row.dateofyear,row.latitude,row.longitude,row.elevation,row.tmax]

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
    StructField('dateofyear', FloatType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
    StructField('label', IntegerType(), False),
])


print("Reading input data...")
data = spark.read.csv(input_file,sep=' ',schema=schema)

data = data.drop('station')

data = data.na.drop()

train_data,test_data = data.randomSplit([0.8,0.2]) # Splitting the dataset between training and testing data

######
#### Start of hyper parameter search using Scikit Learn
######

# Generate the RDD of hyperparameters to test
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

# RDD with (model,accuracy)
start = time.time()
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

#######
#### End of hyper parameter search
#######

data1, data2, data3 = train_data.randomSplit([1.0,2.0,20.0],1234) # Splitting thetraining data into 3 subsets for boosting, each used for one MLP. The second and third sets are twice and ten times larger than the first, respectively

assembler = VectorAssembler(inputCols=['dateofyear','latitude', #vectorizing the input features(required for Spark ML)
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')

trainer = MultilayerPerceptronClassifier(maxIter=iters, 
                                         layers=layers,
                                         stepSize=lr,
                                         blockSize=128, 
                                         seed=1234)

pipeline = Pipeline(stages=[assembler,trainer])

print("Generating and training experts...")
start = time.time()
#Training the first MLP
model1 = pipeline.fit(data1) # Using the first subset to train the first MLP

predictions = model1.transform(data2) # Testing the first MLP on the second data subset

### Generating the training dataset for the second MLP by selecting half of the correct and half of the incorrect predictions made by the first MLP
data2_1 , data2_2 = predictions.randomSplit([0.5,0.5],1234) 

data2_1 = data2_1.select('*').where(data2_1['label']==data2_1['prediction']).drop('features','prediction') # Correct predictions

data2_2 = data2_2.select('*').where(data2_2['label']!=data2_2['prediction']).drop('features','prediction') #Incorrect predictions

train2 = data2_1.union(data2_2) #This is the second training data set

model2 = pipeline.fit(train2) # Training the second MLP

predictions = model2.transform(data2) # Testing the first MLP on the second data subset

#Generating the training dataset for the third MLP by selecting the contradictory predictions made by MLP1 and MLP2 on the third data subset
predictions1 = model1.transform(data3)
predictions2 = model2.transform(data3)
predictions1 = predictions1.select(predictions1['features'],predictions1['prediction'].alias('prediction1'))
train3 = predictions2.join(predictions1,'features')
train3 = train3.select('dateofyear','latitude','longitude','elevation','tmax','label').where(train3['prediction']!=train3['prediction1'])

model3 = pipeline.fit(train3) #Training the third MLP

#Generating predictions from the three MLPs on the test data (without boosting)
print("Generating predictions...")
predictions1 = model1.transform(test_data)
predictions2 = model2.transform(test_data)
predictions3 = model3.transform(test_data)

end = time.time()
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

#Evaluating the three MLPs on the test data (without boosting)
score1 = evaluator.evaluate(predictions1)
score2 = evaluator.evaluate(predictions2)
score3 = evaluator.evaluate(predictions3)

#Generating the ensemble prediction by using the majority vote from the three MLPs
predictions1 = predictions1.select(predictions1['features'],predictions1['prediction'].alias('prediction1'))
predictions2 = predictions2.select(predictions2['features'],predictions2['prediction'].alias('prediction2'))
predictions3 = predictions3.join(predictions1,'features').join(predictions2,'features')
predictions3 = predictions3.withColumn('prediction',predictions3['prediction']+predictions3['prediction1']+predictions3['prediction2']).drop('prediction1','prediction2')

ensemblePrediction = predictions3.withColumn('prediction',predictions3['prediction'] >= 2)
ensemblePrediction = ensemblePrediction.withColumn('prediction',ensemblePrediction['prediction'].cast(DoubleType()))

score = evaluator.evaluate(ensemblePrediction)
TP = ensemblePrediction.where(ensemblePrediction.label==1).where(ensemblePrediction.prediction==1).count()
TN = ensemblePrediction.where(ensemblePrediction.label==0).where(ensemblePrediction.prediction==0).count()
FP = ensemblePrediction.where(ensemblePrediction.label==0).where(ensemblePrediction.prediction==1).count()
FN = ensemblePrediction.where(ensemblePrediction.label==1).where(ensemblePrediction.prediction==0).count()
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print("---------------------- Final train/test info ----------------------")
print("Accuracy for first expert :"+str(score1))
print("Accuracy for second expert :"+str(score1))
print("Accuracy for third expert :"+str(score1))
print("Final accuracy : "+str(score))
print("Confusion Matrix:")
print("TP=",TP," FN=",FN)
print("FP=",FP," TN=",TN)
print("Precision:",precision)
print("Recall:",recall)
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------------")
 

