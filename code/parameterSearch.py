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
	return [row._c1,row._c2,row._c3,row._c4,row._c5]

def transformTest(row) :
	return row._c6

def generateModels(params) :
	model =  MLPClassifier(solver='sgd', learning_rate='constant',
					activation=params[0],
					learning_rate_init=params[1],
					max_iter=params[2],
					hidden_layer_sizes=params[3])
	model.fit(rdd_train_X.value,rdd_train_y.value)
	preds = model.predict(rdd_test_X.value)
	return (model,accuracy_score(rdd_test_y.value,preds))

# 
def getBestModel(m1,m2) :
	if (m1[1] > m2[1]) :
		return m1
	else :
		return m2
 
if __name__ == "__main__":

	# Generate the RDD of hyperparameters to test
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
	weatherData = spark.read.format("com.databricks.spark.csv") \
	.option("header", "false").option("inferSchema", "true") \
	.option("delimiter", ' ').load(inputData)

	# Drop station column
	weatherData = weatherData.drop(weatherData._c0)

	# Randomly split data for training and validation
	train,test = weatherData.randomSplit([0.8, 0.2])

	# Generate dataframes for classlabels and drop class rows
	trainY = train.select(train._c6)
	testY = test.select(test._c6)
	train = train.drop(train._c6)
	test = test.drop(test._c6)

	# Transform dataframes into rdds
	trainRdd = train.rdd 
	testRdd = test.rdd 
	trainYRdd = trainY.rdd 
	testYRdd = testY.rdd

	# Generate train and test Rdds with rows as lists
	trainRdd = trainRdd.map(listTransformTrain)
	testRdd = testRdd.map(listTransformTrain)

	# Generate train and test class labels as singletons
	trainYRdd = trainYRdd.map(transformTest)
	testYRdd = testYRdd.map(transformTest)

	# Broadcast train data and train labels
	rdd_train_X = sc.broadcast(trainRdd.collect())
	rdd_train_y = sc.broadcast(trainYRdd.collect())

	# # Broadcast test data
	rdd_test_X = sc.broadcast(testRdd.collect())
	rdd_test_y = sc.broadcast(testYRdd.collect())

	# RDD with (model,accuracy)
	modelsRdd = paramsRdd.map(generateModels)

	# Get the model with best accuracy :
	bestModel = modelsRdd.reduce(getBestModel)

	# # Print best model and accuracy
	print(bestModel)
	print("Model Info :")
	print("Activation function :"+bestModel[0].activation)
	print("Max iters : "+str(bestModel[0].max_iter))
	print("Learning rate : "+str(bestModel[0].learning_rate_init))
	print("Num hidden layers : "+str(len(bestModel[0].hidden_layer_sizes)))
	print("Hidden layer numbers : "+str(bestModel[0].hidden_layer_sizes))
	print("Accuracy : "+str(bestModel[1]))