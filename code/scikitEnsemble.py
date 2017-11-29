from pyspark import SparkConf, SparkContext
import sys
import operator
import math
import re, string
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from datetime import datetime
import numpy as np
import time

# Input file name
inputData = sys.argv[1]

spark = SparkSession.builder.appName('Parameter search').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+
 
# Generates a hyperparameter RDD with the indicated parameters
def generateHyperParamsRDD() :
	# Use this for long tests
	# activationFuncs = ['logistic', 'tanh', 'relu']
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
	return [row._c1,row._c2,row._c3,row._c4,row._c5]

def listTransformTrainWithLabel(row) :
	return [row._c1,row._c2,row._c3,row._c4,row._c5,row._c6]

def transformTest(row) :
	return row._c6

# Generate all the models with the different hyperparameter combinations
def generateModels(params) :
	model =  MLPClassifier(solver='sgd', learning_rate='constant',
					activation=params[0],
					learning_rate_init=params[1],
					max_iter=params[2],
					hidden_layer_sizes=params[3])
	model.fit(rdd_train_X.value,rdd_train_y.value)
	preds = model.predict(rdd_val_X.value)
	return (model,accuracy_score(rdd_val_y.value,preds))

# Obtain the best model out of the different ones based on accuracy
def getBestModel(m1,m2) :
	if (m1[1] > m2[1]) :
		return m1
	else :
		return m2
 
# Train the final models to use in the ensemble
def trainFinalModels(trainList) :
	# Change to numpy representation
	npTrainList = np.array(trainList)

	# Get class Labels
	trainY = npTrainList[:,5]
	trainSet = np.delete(npTrainList,5,1)

	# Create and train the model 
	model =  MLPClassifier(solver='sgd', learning_rate='constant',
				activation=act,
				learning_rate_init=lr,
				max_iter=iters,
				hidden_layer_sizes=hlayers)
	model.fit(trainSet,trainY)
	return model

def getTestPredictions(model) :
	return model.predict(rdd_test_X.value)


def averagePreds(p1,p2) :
	newList = []

	for i in range(len(p1)) :
		newList.append(p1[i]+p2[i])

	npNewList = np.array(newList)

	return npNewList

def calcPreds(pred) :
	newPred = pred / 10
	if newPred >= 0.5 :
		newPred = 1
	else :
		newPred = 0
	return newPred

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

# 	return preds
if __name__ == "__main__":

	# Generate the RDD of hyperparameters to test
	print("Filling Hyperparameter RDD...")
	paramsRdd = generateHyperParamsRDD()

	schema = StructType([
		StructField('station', StringType(), False), \
		StructField('dateofyear', FloatType(), False), \
		StructField('latitude', FloatType(), False), \
		StructField('longitude', FloatType(), False), \
		StructField('elevation', FloatType(), False), \
		StructField('tmax', FloatType(), False), \
		StructField('value', IntegerType(), False)
	])

	# weatherData = spark.read.csv(inputData,schema,sep=' ')
	print("Reading input data...")
	weatherData = spark.read.format("com.databricks.spark.csv") \
	.option("header", "false").option("inferSchema", "true") \
	.option("delimiter", ' ').load(inputData)

	# Drop station column
	weatherData = weatherData.drop(weatherData._c0)

	# Randomly split data for training and test
	realTrain,test = weatherData.randomSplit([0.8, 0.2])

	# Here is where the hyperparameter search starts

	# Divide the training data into train and validation
	# for hyperparameter search
	train,val = realTrain.randomSplit([0.8,0.2])

	# Generate dataframes for classlabels and drop class rows
	trainY = train.select(train._c6)
	valY = val.select(val._c6)
	train = train.drop(train._c6)
	val = val.drop(val._c6)

	# Transform dataframes into rdds
	trainRdd = train.rdd 
	valRdd = val.rdd 
	trainYRdd = trainY.rdd 
	valYRdd = valY.rdd

	# Generate train and val Rdds with rows as lists
	trainRdd = trainRdd.map(listTransformTrain)
	valRdd = valRdd.map(listTransformTrain)

	# Generate train and val class labels as singletons
	trainYRdd = trainYRdd.map(transformTest)
	valYRdd = valYRdd.map(transformTest)

	# Broadcast train data and train labels
	rdd_train_X = sc.broadcast(trainRdd.collect())
	rdd_train_y = sc.broadcast(trainYRdd.collect())

	# # Broadcast val data
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

	# Broadcast best hyperparameters
	sc.broadcast(act)
	sc.broadcast(iters)
	sc.broadcast(lr)
	sc.broadcast(hlayers)

	# Measure time
	end = time.time()

	print("---------------------- Best model info ----------------------")
	print("Activation func : "+act)
	print("Max epochs : "+str(iters))
	print("Learning rate : "+str(lr))
	print("Hidden layers : " + str(hlayers))
	print("Time : "+str(end - start)+" seconds")
	print("-------------------------------------------------------------")

	# Here is where the hyperparameter search ends 

	# Generate the 10 data sets, one for each expert in the ensemble
	print("Processing data for final training and testing...")
	data1,data2,data3,data4,data5, \
	data6,data7,data8,data9,data10 = realTrain. \
	randomSplit([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

	# Obtain test labels and set
	testY = test.select(test._c6)
	test = test.drop(test._c6)

	# Generate test rdd
	testYRdd = testY.rdd 
	testRdd = test.rdd

	# Transform from rows into lists
	testYRdd = testYRdd.map(transformTest)
	testRdd = testRdd.map(listTransformTrain)

	# Broadcast test set and labels
	rdd_test_X = sc.broadcast(testRdd.collect())
	rdd_test_y = sc.broadcast(testYRdd.collect())

	# Turn all of the training sets into RDDs
	# (these still have their respective class labels)
	data1Rdd = data1.rdd.map(listTransformTrainWithLabel)
	data2Rdd = data2.rdd.map(listTransformTrainWithLabel)
	data3Rdd = data3.rdd.map(listTransformTrainWithLabel)
	data4Rdd = data4.rdd.map(listTransformTrainWithLabel)
	data5Rdd = data5.rdd.map(listTransformTrainWithLabel)
	data6Rdd = data6.rdd.map(listTransformTrainWithLabel)
	data7Rdd = data7.rdd.map(listTransformTrainWithLabel) 
	data8Rdd = data8.rdd.map(listTransformTrainWithLabel) 
	data9Rdd = data9.rdd.map(listTransformTrainWithLabel) 
	data10Rdd = data10.rdd.map(listTransformTrainWithLabel)

	# Make a list of each Rdd list
	rddList = [data1Rdd.collect(),data2Rdd.collect(),data3Rdd.collect(), \
	data4Rdd.collect(),data5Rdd.collect(),data6Rdd.collect(),data7Rdd.collect(), \
	data8Rdd.collect(),data9Rdd.collect(),data10Rdd.collect()]

	# Make an rdd for each training list
	trainRdd = sc.parallelize(rddList)

	# Measure time
	start = time.time()

	# Generate an Rdd of trained models
	print("Training experts...")
	modelRdd = trainRdd.map(trainFinalModels)

	# Get predictions
	print("Generating predictions...")
	predRdd = modelRdd.map(getTestPredictions)

	# Sum all predictions
	sumPreds = predRdd.reduce(averagePreds)

	# Create an Rdd of the predictions
	sumPredsRdd = sc.parallelize(sumPreds)

	# Get final predictions by getting the average and transforming to 1s and 0s
	finalPreds = sumPredsRdd.map(calcPreds)

	# Calculate final accuracy
	finalAccuracy = accuracy_score(testYRdd.collect(),finalPreds.collect())

	# Generate confusion matrix
	TP,TN,FP,FN = genConfMatrix(testYRdd.collect(),finalPreds.collect())

	# Measure time
	end = time.time()
	# Print accuracy

	print("---------------------- Final train/test info ----------------------")
	print("Final accuracy : "+str(finalAccuracy))
	print("Real 1s classified as 1s : "+str(TP)) 
	print("Real 0s classified as 0s : "+str(TN)) 
	print("Real 1s classified as 0s : "+str(FN))
	print("Real 0s classified as 1s : "+str(FP))
	print("Time : "+str(end - start)+" seconds")
	print("-------------------------------------------------------------------")



	

