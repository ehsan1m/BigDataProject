from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import time

spark = SparkSession.builder.appName('Boosted MLPs').getOrCreate()

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

data = data.drop('station')

data = data.na.drop()

train_data,test_data = data.randomSplit([0.8,0.2]) # Splitting the dataset between training and testing data

data1, data2, data3 = train_data.randomSplit([1.0,2.0,20.0],1234) # Splitting the training data into 3 subsets for boosting, each used for one MLP. The second and third sets are twice and twenty times larger than the first, respectively

assembler = VectorAssembler(inputCols=['dateofyear','latitude', #vectorizing the input features(required for Spark ML)
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')


##### GRID PARAMETER BUILDER

mlpc = MultilayerPerceptronClassifier()

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
print("Creating parameter grid builder...")
paramGrid = ParamGridBuilder() \
	.addGrid(mlpc.maxIter, [500,1000]) \
    .addGrid(mlpc.layers, [[5,1,2],[5,2,2],[5,5,2]])\
    .addGrid(mlpc.stepSize, [0.5,0.2,0.1,0.05,0.02])\
    .addGrid(mlpc.solver, ['l-bfgs'])\
    .addGrid(mlpc.tol, [1e-06])\
    .build()

# SIMPLER COMBINATION FOR TEST
# paramGrid = ParamGridBuilder().addGrid(mlpc.maxIter, [5, 10]) \
#     .addGrid(mlpc.layers, [[5,2,2],[5,5,2]])\
#     .addGrid(mlpc.stepSize, [0.5,0.2])\
#     .addGrid(mlpc.solver, ['l-bfgs', 'gd'])\
#     .addGrid(mlpc.tol, [1e-06, 1e-05])\
#     .build()

# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
print("Calculating best model...")
start = time.time()
tvs = TrainValidationSplit(estimator=mlpc,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.

train_data_grid_search = (assembler.transform(train_data)).select('features','label')

model = tvs.fit(train_data_grid_search)

# Save the parameters of the best model into variables
bestmodel = model.bestModel
layers = list(bestmodel._java_obj.parent().getLayers())
iters = bestmodel._java_obj.parent().getMaxIter()
solver = bestmodel._java_obj.parent().getSolver()
tol = bestmodel._java_obj.parent().getTol()
lr = bestmodel._java_obj.parent().getStepSize()

end = time.time()

print("---------------------- Best model info ----------------------")
print("Max epochs : "+str(iters))
print("Learning rate : "+str(lr))
print("Hidden layers : " + str(layers))
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------")


#### End of grid parameter search


#layers = [5, 15, 2]

trainer = MultilayerPerceptronClassifier(maxIter=iters, 
                                             layers=layers,
                                             stepSize=lr,
                                             blockSize=128,
                                             solver=solver, ##### include in the other version
                                             tol=tol, #### include in the other version
                                             seed=1234)

pipeline = Pipeline(stages=[assembler,trainer])

#Training the first MLP
print("Generating and training experts...")
start = time.time()
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
print("TP FN")
print("FP TN")
print("TP=",TP," FN=",FN)
print("FP=",FP," TN=",TN)
print("Precision:",precision)
print("Recall:",recall)
print("Time : "+str(end - start)+" seconds")
print("-------------------------------------------------------------------")
