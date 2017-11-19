from pyspark import SparkConf, SparkContext
import sys
import operator
import math
import re, string
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
conf = SparkConf().setAppName('Parameter search')
sc = SparkContext(conf=conf)
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.2'  # make sure we have Spark 2.2+
 
def generateModels(params) :
	model =  MLPClassifier(solver='sgd', learning_rate='constant',
					learning_rate_init=params[0],
					max_iter=params[1],
                    hidden_layer_sizes=params[2])
	model.fit(rdd_train_X.value,rdd_train_y.value)
	preds = model.predict(rdd_test_X.value)
	return (model,accuracy_score(rdd_test_y.value,preds))

def getBestModel(m1,m2) :
	if (m1[1] > m2[1]) :
		return m1
	else :
		return m2

# Function to add up 2 data points.
def addPoints(p1,p2) :
	return (p1[0]+p2[0],p1[1]+p2[1])

# Format of the output
def output_format(kv):
    v1,v2 = kv
    return 'r = %f \nr^2 = %f' % (v1,v2)
 
if __name__ == "__main__":

	# Solvers = 'adam','sgd','lbfgs' >> need sgd to use learning rate
	# Alpha is L2 regularization term.
	# Use learning_rate 'constant' and then learning_rate_init to define the number
	# max_iter
	# Hyperparameters to test :
	# Different learning rate, max_iters,number of layers and number of neurons per layer
	learnRates = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
	maxIters = [50,100,200,500,1000,2000]
	numHiddenL = [1,2,3]
	neuronsPerLayer = [1,2,3,4,5]
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
	for lr in learnRates :
		for iters in maxIters :
			for hl in hiddenLayerNums :
				hyperParams.append([lr,iters,hl])


	# Transform the hyperparameter array into an RDD
	paramsRdd = sc.parallelize(hyperParams)

	# Broadcast train data and train labels
	X = [[0., 0.], [1., 1.]]
	rdd_train_X = sc.broadcast(X)
	y = [0, 1]
	rdd_train_y = sc.broadcast(y)

	# Broadcast test data
	t = [[2., 2.], [-1., -2.]]
	rdd_test_X = sc.broadcast(t)
	tY = [0, 1]
	rdd_test_y = sc.broadcast(tY)

	# RDD with (model,accuracy)
	modelsRdd = paramsRdd.map(generateModels)

	# Get the model with best accuracy :
	bestModel = modelsRdd.reduce(getBestModel)

	# Print best model and accuracy
	print(bestModel)